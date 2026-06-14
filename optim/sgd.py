from lantern.optim.optimizer import Optimizer
import torch

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
                if weight_decay != 0.0:
                    grad = grad.add(p, alpha=weight_decay)

                # First moment
                if momentum != 0.0:
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
