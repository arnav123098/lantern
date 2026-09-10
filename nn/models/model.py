import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def num_params(self, grad=False):
        if grad:
            return sum([p.numel() for p in self.parameters() if p.requires_grad])
        else:
            return sum([p.numel() for p in self.parameters()])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise NotImplementedError

    def mfu(self, *args, **kwargs):
        raise NotImplementedError
