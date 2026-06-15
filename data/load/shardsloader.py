from lantern.data.load.dataloader import DataLoader
from lantern.tokenizer import Tokenizer
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
        batch_size: int,
        block_size: int,
        split: list[str],
        repo_id: str,
        tokenizer: Tokenizer
    ):
        super().__init__()

        self.B = batch_size
        self.T = block_size

        # dataloading
        self.shards = split # a list of shard filepaths
        self.tokenized_shards = [f.replace('.parquet', '.npy').replace(repo_id, f'{repo_id}_tokenized') for f in self.shards]

        os.makedirs(f'{self.datasets.get_path(repo_id)}_tokenized', exist_ok=True)

        # pre-tokenize (if not already)
        self.tokenizer = tokenizer

        self.eos = self.tokenizer.eos_token_id
        if self.eos is None: self.eos = self.tokenizer.encode('\n\n')[0] # fallback to newlines

        self.tokenize_shards()

        self.n_shards = len(self.shards)

        self.ptr = 0
        self.shard_ptr = 0
        
        self.curr_shard_tokens = np.load(self.tokenized_shards[0])

        self.state_keys = ('ptr', 'shard_ptr')

    def tokenize_shards(self):
        for shard, tokenized_path in zip(self.shards, self.tokenized_shards):
          if self.datasets.exists(tokenized_path): continue

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

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        leftover = np.array([], dtype=np.uint32)
        remaining = self.B * self.T

        step = remaining + 1
        end_pos = self.ptr + step

        # check bounds within shard
        if self.ptr + step > len(self.curr_shard_tokens):
            leftover = self.curr_shard_tokens[self.ptr:]
            remaining -= len(leftover)

            step = remaining + 1
            end_pos = step

            self.shard_ptr += 1
            self.ptr = 0
            self.curr_shard_tokens = None

        # check bounds over the dataset
        if self.shard_ptr >= self.n_shards:
          buf = leftover if len(leftover) > 0 else None
        else:
          if self.curr_shard_tokens is None:
            self.curr_shard_tokens = np.load(self.tokenized_shards[self.shard_ptr])
          buf = np.concatenate([
              leftover,
              self.curr_shard_tokens[self.ptr:end_pos]
          ], dtype=np.uint32)

          # update pos
          self.ptr = end_pos

        if buf is not None:
          if len(buf) < self.B * self.T + 1:
            return None, None # change this later to loop over the dataset for multiple epochs
          # make B x T tensors
          X = torch.from_numpy(buf[:-1]).view(self.B, self.T)
          Y = torch.from_numpy(buf[1:]).view(self.B, self.T)
        else:
          return None, None

        return X.long(), Y.long()
