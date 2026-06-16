from lantern.data.load.dataloader import DataLoader
from lantern.tokenizer import Tokenizer
import os
import numpy as np
import pyarrow.parquet as pq
import torch

'''
This loader will be used to load large datasets like FineWeb or CommonCrawl.
TODO (for later): batch_encode to speed up tokenization 
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
        self.n_shards = len(self.shards)
        assert self.n_shards > 0, "No shards found"

        self.tokenized_shards = [f.replace('.parquet', '.bin').replace(repo_id, f'{repo_id}_tokenized') for f in self.shards]

        os.makedirs(f'{self.datasets.get_path(repo_id)}_tokenized', exist_ok=True)

        # pre-tokenize (if not already)
        self.tokenizer = tokenizer

        self.eos = self.tokenizer.eos_token_id
        if self.eos is None: self.eos = self.tokenizer.encode('\n\n')[0] # fallback to newlines

        self.tokenize_shards()

        self.epoch = 0

        self.ptr = 0
        self.shard_ptr = 0
        
        np.random.shuffle(self.tokenized_shards)
        self.curr_shard_tokens = None

        self.state_keys = ('ptr', 'shard_ptr', 'epoch', 'tokenized_shards')

    def tokenize_shards(self):
        for shard, tokenized_path in zip(self.shards, self.tokenized_shards):
          if self.datasets.exists(tokenized_path): continue

          pfile = pq.ParquetFile(shard)
          
          for rg in range(pfile.num_row_groups):
            tokens = []
            text_content = pfile.read_row_group(rg, columns=['text'])['text']
            for text in text_content:
              ids = self.tokenizer.encode(
                      text.as_py()
              )

              ids.append(self.eos)

              tokens.extend(ids)

            tokens = np.array(tokens, dtype=np.uint32)
            
            with open(tokenized_path, 'ab') as f:
              tokens.tofile(f)

        print('Shards tokenized successfully')

    def _next_shard(self):
        self.shard_ptr += 1
        self.ptr = 0

        if self.shard_ptr >= self.n_shards:
            self.epoch += 1
            self.shard_ptr = 0
            np.random.shuffle(self.tokenized_shards) 

        self.shard_ptr = self.shard_ptr % self.n_shards
        self.curr_shard_tokens = np.fromfile(self.tokenized_shards[self.shard_ptr], dtype=np.int32)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        if self.curr_shard_tokens is None:
           self.curr_shard_tokens = np.fromfile(self.tokenized_shards[self.shard_ptr], dtype=np.uint32)

        buf_parts = []
        needed = self.B * self.T + 1

        while needed > 0:
            available = len(self.curr_shard_tokens) - self.ptr

            if available == 0:
                self._next_shard()
                continue
            
            take = min(needed, available)

            buf_parts.append(
                self.curr_shard_tokens[self.ptr:self.ptr + take]
            )

            self.ptr += take
            needed -= take

        buf = np.concatenate(buf_parts, dtype=np.uint32)
        assert len(buf) == self.B * self.T + 1

        X = torch.from_numpy(buf[:-1]).view(self.B, self.T)
        Y = torch.from_numpy(buf[1:]).view(self.B, self.T)

        return X.long(), Y.long()
