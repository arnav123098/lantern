import torch.nn as nn

'''
This file will contain common utilities for making stuff as well as for running experiments.
'''

class dotdict(dict):
    # dot.notation access to dictionary attributes
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def get_optim_groups(model: nn.Module, weight_decay: float | None = None):
    decay = []
    no_decay = []

    for p in model.parameters():
        if not p.requires_grad:
            continue

        if p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)

    decay_params = {"params": decay}
    if weight_decay is not None:
        decay_params["weight_decay"] = weight_decay

    return [
        decay_params,
        {"params": no_decay, "weight_decay": 0.0}
    ]
