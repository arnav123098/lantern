from lantern.data.load.dataloader import DataLoader
from lantern.tokenizer import Tokenizer
import os
import numpy as np
import pyarrow.parquet as pq
import torch

'''
This loader will be used to load large datasets like FineWeb or CommonCrawl.
'''
class ShardsLoader(DataLoader):
    def __init__(
        self,
        batch_size: int,
        block_size: int,
        split: list[str],
        repo_id: str,
        tokenizer: Tokenizer,
        tok_batch_size: int = 256, # doesn't matter if tokenizer.has_batch_encoding is False
        rank: int = 0,
        world_size: int = 1
    ):
        super().__init__()

        self.rank = rank
        self.world_size = world_size

        self.B = batch_size
        self.T = block_size

        # dataloading
        self.shards = split # a list of shard filepaths
        self.n_shards = len(self.shards)
        assert self.n_shards > 0, "No shards found"

        self.tokenized_shards = [f.replace('.parquet', '.bin').replace(repo_id, f'{repo_id}_tokenized') for f in self.shards]

        os.makedirs(f'{self.datasets.get_path(repo_id)}_tokenized', exist_ok=True)

        # pre-tokenize (if not already)
        self.tokenizer = tokenizer

        self.eos = self.tokenizer.eos_token_id

        self.tokenize_shards(tok_batch_size if self.tokenizer.has_batch_encoding else 1)

        self.epoch = 0

        self.ptr = self.B * self.T * self.rank
        self.shard_ptr = 0

        np.random.shuffle(self.tokenized_shards)
        self.curr_shard_tokens = None

        self.state_keys = ('ptr', 'shard_ptr', 'epoch', 'tokenized_shards')

    def tokenize_shards(self, tok_batch_size):
        for shard, tokenized_path in zip(self.shards, self.tokenized_shards):
          if self.datasets.exists(tokenized_path): continue

          print(f'Tokenizing shard: {shard}')

          pfile = pq.ParquetFile(shard)

          encode = self.tokenizer.encode_batch if self.tokenizer.has_batch_encoding else self.tokenizer.encode

          with open(tokenized_path, 'wb') as f:
            for rg in range(pfile.num_row_groups):
              text_content = pfile.read_row_group(rg, columns=['text'])['text']

              for i in range(0, len(text_content), tok_batch_size):
                batch = [
                          text.as_py()
                          for text in text_content[i:i + tok_batch_size]
                      ]
      
                encoded = encode(batch[0] if tok_batch_size == 1 else batch)

                tokens = []

                for ids in encoded:
                    ids = list(ids)
                    if self.eos is not None:
                      ids.append(self.eos)
                    tokens.extend(ids)

                np.asarray(tokens, dtype=np.uint32).tofile(f)
                    
        print('Shards tokenized successfully')

    def _next_shard(self):
        self.shard_ptr += 1
        self.ptr = self.B * self.T * self.rank

        if self.shard_ptr >= self.n_shards:
            self.epoch += 1
            self.shard_ptr = 0
            np.random.shuffle(self.tokenized_shards)

        self.shard_ptr = self.shard_ptr % self.n_shards
        self.curr_shard_tokens = np.memmap(
            self.tokenized_shards[self.shard_ptr],
            dtype=np.uint32,
            mode='r'
        )

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        if self.curr_shard_tokens is None:
            self.curr_shard_tokens = np.memmap(
                self.tokenized_shards[self.shard_ptr],
                dtype=np.uint32,
                mode='r'
            )

        assert len(self.curr_shard_tokens) > self.B * self.T * self.rank

        buf_parts = []
        needed = self.B * self.T + 1

        while needed > 0:
            available = len(self.curr_shard_tokens) - self.ptr

            if available <= 0:
                self._next_shard()
                continue

            take = min(needed, available)

            buf_parts.append(
                self.curr_shard_tokens[self.ptr:self.ptr + take]
            )

            self.ptr += take
            needed -= take

        buf = np.concatenate(buf_parts, dtype=np.uint32)

        self.ptr += self.B * self.T * (self.world_size - 1)

        assert len(buf) == self.B * self.T + 1

        X = torch.from_numpy(buf[:-1]).view(self.B, self.T)
        Y = torch.from_numpy(buf[1:]).view(self.B, self.T)

        return X.long(), Y.long()
