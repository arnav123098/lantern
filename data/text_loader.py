import tiktoken
import torch
from dataloader import DataLoader

'''
The simplest dataloader: load text from a file and create train and val splits; track current position and update when returning a B x T tensor as the current batch.
'''
class TextLoaderLite(DataLoader):
    def __init__(
        self,
        config
    ):
        super().__init__()

        self.B = config.batch_size if config.B is None else config.B
        self.T = config.block_size

        self.val_B = config.val_batch_size

        # load tokens
        with open(config.filepath, 'r') as f:
            text = f.read()
        tokens = config.tokenizer.encode(text)
        self.tokens = torch.tensor(tokens)
        train_split_len = int(self.tokens.size(-1) * (1 - config.val_split))

        # create splits
        self.train = self.tokens[:train_split_len]
        self.val = self.tokens[train_split_len:]

        self.train_len = len(self.train)
        self.val_len = len(self.val)

        self.train_curr_pos = 0
        self.val_curr_pos = 0

        self.state_keys = ('train_curr_pos', 'val_curr_pos')

    def next_train_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        step = self.B * self.T + 1
        end_pos = self.train_curr_pos + step
        
        # check bounds
        if self.train_curr_pos + step > self.train_len:
            self.train_curr_pos = 0
            end_pos = step
        
        buf = self.train[self.train_curr_pos:end_pos]

        # update pos
        self.train_curr_pos = end_pos

        # make B x T tensors
        train_X = buf[:-1].view(self.B, self.T)
        train_Y = buf[1:].view(self.B, self.T)
        

        return train_X, train_Y

    def next_val_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        step = self.val_B * self.T + 1
        end_pos = self.val_curr_pos + step

        if self.val_curr_pos + step > self.val_len:
            self.val_curr_pos = 0
            end_pos = step

        buf = self.val[self.val_curr_pos:end_pos]
        self.val_curr_pos = end_pos

        val_X = buf[:-1].view(self.val_B, self.T)
        val_Y = buf[1:].view(self.val_B, self.T)

        return val_X, val_Y
