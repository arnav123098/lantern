from optimizer import Optimizer
import torch

'''
RMSProp Optimizer tracks the second moment (v) i.e. the uncentered variance, only concerned with the size of the change.
- v = alpha * prev_v + (1 - alpha) * grad
- update = grad / (sqrt(v) + eps) where eps is a small number to prevent div by zero

This helps scale down steps for params with large grad and scale up steps for params with small grad.
'''
class RMSProp(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        defaults = dict(
            lr = lr,
            alpha = alpha,
            eps = eps,
            weight_decay = weight_decay
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None: continue

                grad = p.grad

                # L2 regularization
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state.setdefault(p, {})
                buf = state.get('v_buf')

                if buf is None:
                    buf = state['v_buf'] = grad.pow(2)
                else:
                    buf.mul_(alpha).addcmul_(grad, grad, value=(1 - alpha))

                update = grad / (buf.sqrt().add_(eps))
                p.add_(update, alpha=-lr)
