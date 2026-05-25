import tiktoken
import torch

'''
The simplest dataloader: load text from a file and create train and val splits; track current position and update when returning a B x T tensor as the current batch.
'''
class TextLoaderLite:
    def __init__(
        self,
        batch_size: int,
        block_size: int,
        filepath: str,
        model: str,
        val_split: int=0
    ):
        self.B = batch_size
        self.T = block_size

        # load tokens
        with open(filepath, 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding(model)
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        train_split_len = self.tokens.size(-1) * (1 - val_split)

        # create splits
        self.train = self.tokens[:train_split_len]
        self.val = self.tokens[train_split_len:]

        self.train_len = len(self.train)
        self.val_len = len(self.val)

        self.train_curr_pos = 0
        self.val_curr_pos = 0

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        step = self.B * self.T + 1
        train_end_pos, val_end_pos = self.train_curr_pos + step, self.val_curr_pos + step
        
        # check bounds
        if self.train_curr_pos + step > self.train_len:
            self.train_curr_pos = 0
            train_end_pos = step
        if self.val_curr_pos + step > self.val_len:
            self.val_curr_pos = 0
            val_end_pos = step
        
        train_buf = self.train[self.train_curr_pos:train_end_pos]
        val_buf = self.val[self.val_curr_pos:val_end_pos]

        # update pos
        self.train_curr_pos, self.val_curr_pos = train_end_pos, val_end_pos

        # make B x T tensors
        train_X = train_buf[:-1].view(self.B, self.T)
        train_Y = train_buf[1:].view(self.B, self.T)
        val_X = val_buf[:-1].view(self.B, self.T)
        val_Y = val_buf[1:].view(self.B, self.T)

        return train_X, train_Y, val_X, val_Y
