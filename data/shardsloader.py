from lantern.utils import dotdict
from lantern.data.dataloader import DataLoader
import os
import numpy as np
import pyarrow.parquet as pq
import torch

'''
This loader will be used to load large datasets like FineWeb or CommonCrawl. For now it works, and actually works well. But I've only made the train batch loader since I'm gonna organize stuff for sometime and make things cleaner. So I'm marking this as incomplete for now.
'''
class ShardsLoader(DataLoader):
    def __init__(
        self,
        config: dotdict
    ):
        super().__init__()

        self.downloader.download_dataset(config.repo_id, config.max_shards)

        self.B = config.batch_size if config.B is None else config.B
        self.T = config.block_size

        self.val_B = config.val_batch_size

        # dataloading
        self.shards = self.downloader.get_files(config.repo_id)
        self.tokenized_shards = [f.replace('.parquet', '.npy').replace(config.repo_id, f'{config.repo_id}_tokenized') for f in self.shards]

        os.makedirs(f'{self.downloader.get_path('HuggingFaceFW')}/fineweb_tokenized', exist_ok=True)

        # pre-tokenize (if not already)
        self.tokenizer = config.tokenizer

        # self.eos = self.tokenizer.eos_token_id
        self.eos = self.tokenizer.eot_token # TODO: this works only with tiktoken; make a generic wrapper for tokenizers
        if self.eos is None: self.eos = self.tokenizer.tokenize('\n\n') # fallback to newlines

        self.tokenize_shards()

        # create splits (in groups of shards)
        self.n_shards = len(self.shards)
        self.n_train_shards = int(self.n_shards * (1 - config.val_split))
        self.n_val_shards = self.n_shards - self.n_train_shards

        print(f'Train split: {self.n_train_shards} shards\nVal split: {self.n_val_shards} shards')
  
        self.train = self.tokenized_shards[:self.n_train_shards]
        self.val = self.tokenized_shards[self.n_train_shards:]

        self.train_ptr = 0
        self.val_ptr = 0
        self.train_shard_ptr = 0
        self.val_shard_ptr = 0
        
        self.train_shard_tokens = np.load(self.train[0])
        self.val_shard_tokens = np.load(self.val[0])

        self.state_keys = ('train_ptr', 'val_ptr', 'train_shard_ptr', 'val_shard_ptr')

    def tokenize_shards(self):
        for shard, tokenized_path in zip(self.shards, self.tokenized_shards):
          if self.downloader.exists(tokenized_path): continue

          pfile = pq.ParquetFile(shard)
          text_content = pfile.read_row_group(0, columns=['text'])['text']
          
          tokens = []
          for text in text_content:
            ids = self.tokenizer.encode(
                    text.as_py()
            )

            ids.append(self.eos)

            tokens.extend(ids)

          tokens = np.array(tokens, dtype=np.uint32)
          np.save(
              tokenized_path,
              tokens
          )
        print('Shards tokenized successfully')

    def next_train_batch(self) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        leftover = []
        remaining = self.B * self.T

        step = remaining + 1
        end_pos = self.train_ptr + step

        # check bounds within shard
        if self.train_ptr + step > len(self.train_shard_tokens):
            leftover = self.train_shard_tokens[self.train_ptr:]
            remaining -= len(leftover)

            step = remaining + 1
            end_pos = step

            self.train_shard_ptr += 1
            self.train_ptr = 0
            self.train_shard_tokens = None

        # check bounds over the dataset
        if self.train_shard_ptr >= self.n_train_shards:
          buf = leftover if len(leftover) > 0 else None
        else:
          if self.train_shard_tokens is None:
            self.train_shard_tokens = np.load(self.train[self.train_shard_ptr])
          buf = np.concatenate([
              leftover,
              self.train_shard_tokens[self.train_ptr:end_pos]
          ])

          # update pos
          self.train_ptr = end_pos

        if buf is not None:
          if len(buf) < self.B * self.T + 1:
            return None, None
          # make B x T tensors
          train_X = torch.from_numpy(buf[:-1]).view(self.B, self.T)
          train_Y = torch.from_numpy(buf[1:]).view(self.B, self.T)
        else:
          return None, None

        return train_X, train_Y
