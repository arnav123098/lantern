from optimizer import Optimizer
import torch

'''
Lion optimizer keeps track of first moment but only uses the sign during update so every param gets the same update i.e. the learning rate.
key points:
- L2 regularization is decoupled as in AdamW
- it does beta1 * momentum + (1 - beta1) * grad and computes its sign for update
- the momentum buffer update is done using beta2
'''
class Lion(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.99,
        weight_decay: float = 0.0
    ):
        defaults = dict(
            lr = lr,
            beta1 = beta1,
            beta2 = beta2,
            weight_decay = weight_decay
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None: continue

                grad = p.grad

                # L2 regularization (decoupled)
                if weight_decay != 0.0:
                    p.mul_(1 - lr * weight_decay)

                # First moment
                state = self.state.setdefault(p, {}) # create a state for the param if it doesn't already exist
                buf = state.get('momentum_buffer')

                if buf is None:
                    buf = state['momentum_buffer'] = torch.zeros_like(p) # initialize momentum_buffer as current gradient
                    
                update = torch.sign(beta1 * buf + (1 - beta1) * grad) # only get the sign of momentum
                buf.mul_(beta2).add_(grad, alpha=(1 - beta2))
                
                p.add_(update, alpha=-lr)
