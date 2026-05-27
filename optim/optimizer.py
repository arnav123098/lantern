import torch

'''
The Optimizer base class does the following:
- store and manage param_groups (dicts with info about a param group)
- set defaults such as lr, momentum etc. in groups where it's not user defined
- manage a state to store info about a particular param
'''
class Optimizer: # Base class for optimizers
    def __init__(
        self,
        params, 
        defaults: dict
    ):
        self.defaults = defaults
        self.param_groups = []
        self.state = {}

        if isinstance(params, torch.Tensor):
            raise TypeError("params should be an iterable of torch.Tensors or dicts")
        
        for group in params: self._add_param_group(group)
    
    def _add_param_group(self, group):
        for k, v in self.defaults.items():
            group.setdefault(k, v)

        self.param_groups.append(group)

    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.zero_()

    def step(self): raise NotImplementedError
