from optimizer import Optimizer
import torch

'''
AdamW adds one critical fix not in Adam or RMSProp.
- L2 regularization is meant to perform grad = grad + weight_decay * p
- Squaring this updated grad in second moment amplifies the weight_decay producing unwanted behavior
- The simple fix is to apply the L2 regularization separately
'''
class AdamW(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.99,
        eps: float = 1e-8,
        amsgrad: bool = False,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr = lr,
            beta1 = beta1,
            beta2 = beta2,
            eps = eps,
            amsgrad = amsgrad,
            weight_decay = weight_decay
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']
            amsgrad = group['amsgrad']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None: continue

                grad = p.grad

                state = self.state.setdefault(p, {})

                # update current step
                t = state.setdefault('step', 0)
                t = state['step'] = t + 1

                # L2 regularization (decoupled; directly updates the param now)
                if weight_decay != 0.0:
                    p.mul_(1 - lr * weight_decay)

                # update first moment
                m_buf = state.setdefault('m_buf', torch.zeros_like(p))
                m_buf.mul_(beta1).add_(grad, alpha=(1 - beta1))

                # update second moment
                v_buf = state.setdefault('v_buf', torch.zeros_like(p))
                v_buf.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))

                bias_corr_v = 1 - beta2 ** t

                # use max v_buf (theoretical fix for convergence)
                if amsgrad:
                    max_v_buf = state.setdefault('max_v_buf', torch.zeros_like(p))
                    torch.maximum(max_v_buf, v_buf, out=max_v_buf)

                    denom = (max_v_buf / bias_corr_v).sqrt().add_(eps)
                else:
                    denom = (v_buf / bias_corr_v).sqrt().add_(eps)

                # define step size and update params
                bias_corr_m = 1 - beta1 ** t
                step_size = lr / bias_corr_m
                
                p.addcdiv_(m_buf, denom, value=-step_size)
