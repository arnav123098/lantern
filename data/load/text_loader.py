import torch
from lantern.data.load.dataloader import DataLoader

'''
The simplest dataloader: load text from a file and create train and val splits; track current position and update when returning a B x T tensor as the current batch.
'''
class TextLoader(DataLoader):
    def __init__(
        self,
        batch_size: int,
        block_size: int,
        split: list[int]
    ):
        super().__init__()

        self.B = batch_size
        self.T = block_size

        self.tokens = split
        self.n_tokens = len(self.tokens)

        self.curr_pos = 0

        self.state_keys = ('curr_pos')

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        step = self.B * self.T + 1
        end_pos = self.curr_pos + step
        
        # check bounds
        if self.curr_pos + step > self.n_tokens:
            self.curr_pos = 0
            end_pos = step
        
        buf = self.tokens[self.curr_pos:end_pos]

        # update pos
        self.curr_pos = end_pos

        # make B x T tensors
        X = buf[:-1].view(self.B, self.T)
        Y = buf[1:].view(self.B, self.T)
        
        return X, Y
