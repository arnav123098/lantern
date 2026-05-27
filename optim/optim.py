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
        defaults:dict
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

'''
Stochastic Gradient Descent (SGD) Optimizer does the following:
# Basic
- in it's simplest form, sets p -= update * lr where update = grad

# First moment
- with a momentum, update = momentum * previous_momentum + grad
- with an optional dampening, update = momentum * previous_momentum + (1 - dampening) * grad

# Nesterov momentum
- Nesterov momentum; update = momentum * update + grad (i.e. looking one step ahead)
- In theory, the Nesterov grad is the one for the next step.
- Since it requires another forward and backprop, in practice, use the current grad
'''
class SGD(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False
    ):
        defaults = dict(
            lr = lr,
            momentum = momentum,
            weight_decay = weight_decay,
            dampening = dampening,
            nesterov = nesterov
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None: continue

                grad = p.grad

                # L2 regularization
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # First moment
                if momentum != 0:
                    state = self.state.setdefault(p, {}) # create a state for the param if it doesn't already exist
                    buf = state.get('momentum_buffer')

                    if buf is None:
                        buf = state['momentum_buffer'] = grad.clone().detach() # initialize momentum_buffer as current gradient
                    else:
                        buf.mul_(momentum).add_(grad, alpha=(1 - dampening))

                    # Nesterov momentum (to first order accuracy since we use the current grad)
                    if nesterov:
                        update = grad.add(buf, alpha=momentum)
                    else:
                        update = buf
                else: update = grad

                p.add_(update, alpha=-lr)

class RMSProp(Optimizer): pass
