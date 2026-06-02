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

        self.state_keys = ()

    def state_dict(self):
        all_params = [
            p
            for group in self.param_groups
            for p in group['params']
        ]

        state_dict = {
            k: getattr(self, k)
            for k in self.state_keys
        }

        indexed_state = {idx: self.state[p] for idx, p in enumerate(all_params)}
        state_dict['state'] = indexed_state

        return state_dict

    def load_state_dict(self, state_dict):
        state = state_dict['state']

        all_params = [
            p
            for group in self.param_groups
            for p in group['params']
        ]

        assert len(all_params) == len(state), "Params don't match"

        internal_state = {
            p: state[idx] for idx, p in enumerate(all_params) if idx in state
        }
        self.state = internal_state

        del state_dict['state']

        for k, v in state_dict.items():
            setattr(self, k, v)

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
