from lantern.optim.optimizer import Optimizer
import torch

# This file contains both Muon and MuonW optimizers

# TODO: optimizations to make this even faster

'''
Unlike other optimizers we normally come across, Muon focuses on the whole matrix rather than individual params.
This optimizer is focused on overall (matrix) direction of update and removes stretch along different directions ensuring uniformity.

Muon:
- G (gradient matrix) stretches one direction more than others
- Muon removes this
- if x is a vector, length of Gx is |Gx|
    but |Gx|^2 = (Gx)^T(Gx) = (x^T)(G^T)Gx
- This shows that the uneven stretching of x across some dims depends on (G^T)G
- using its eigenvalues and eigenvectors, we can perform Singular Value Decomposition or SVD
    G = UΣ(V^T)
- the update becomes G = U(V^T)
- since this calculation is expensive, Muon implementations generally use Newton-Schulz orthogonalization

(PS: unlike other optimizers, this one's more math and proof heavy so it's recommended to understand its math properly first)
'''
class Muon(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        steps: int = 5,
        beta: float = 0.9,
        nesterov: bool = False,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr = lr,
            beta = beta,
            nesterov = nesterov,
            weight_decay = weight_decay
        )
        super().__init__(params, defaults)

        self.steps = steps
        self.eps = 1e-7

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            nesterov = group['nesterov']
            weight_decay = group['weight_decay']

            for p in group['params']:
                # check if 1D (requires fallback to other optimizers like AdamW)
                if p.grad is None or p.ndim < 2: continue

                grad = p.grad

                state = self.state.setdefault(p, {})

                # update first moment
                m_buf = state.setdefault('m_buf', torch.zeros_like(p))
                x = m_buf.mul_(beta).add_(grad, alpha=(1 - beta))

                # nesterov
                if nesterov:
                    x = grad.add(x, alpha=beta)

                # flatten to 2D (Muon works on matrices only)
                orig_shape = x.shape
                x = x.reshape(orig_shape[0], -1)

                # we don't want square matrix to be the expensive dim, so transpose
                transpose = False
                if x.shape[0] > x.shape[1]:
                    x = x.T
                    transpose = True

                x = x.float() # for numerical stability

                # normalize
                x = x / (x.norm() + self.eps)

                # Newton-Schulz orthogonalization
                for _ in range(self.steps):
                    x = 1.5 * x - 0.5 * (x @ x.T @ x)

                if transpose:
                    x = x.T

                # convert back to original dtype
                x = x.to(p.dtype)

                # reshape
                x = x.reshape(orig_shape)

                # update
                p.add_(x, alpha=-lr)

'''
The only extra thing MuonW has is decoupled weight decay. It's the same as the upgrade from Adam to AdamW.
'''
class MuonW(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        steps: int = 5,
        beta: float = 0.9,
        nesterov: bool = False,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr = lr,
            beta = beta,
            nesterov = nesterov,
            weight_decay = weight_decay
        )
        super().__init__(params, defaults)

        self.steps = steps
        self.eps = 1e-7

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            nesterov = group['nesterov']
            weight_decay = group['weight_decay']

            for p in group['params']:
                # check if 1D (requires fallback to other optimizers like AdamW)
                if p.grad is None or p.ndim < 2: continue

                grad = p.grad

                state = self.state.setdefault(p, {})

                # L2 regularization (decoupled)
                if weight_decay != 0.0:
                    p.mul_(1 - lr * weight_decay)

                # update first moment
                m_buf = state.setdefault('m_buf', torch.zeros_like(p))
                x = m_buf.mul_(beta).add_(grad, alpha=(1 - beta))

                # nesterov
                if nesterov:
                    x = grad.add(x, alpha=beta)

                # flatten to 2D (Muon works on matrices only)
                orig_shape = x.shape
                x = x.reshape(orig_shape[0], -1)

                # we don't want square matrix to be the expensive dim, so transpose
                transpose = False
                if x.shape[0] > x.shape[1]:
                    x = x.T
                    transpose = True

                x = x.float() # for numerical stability

                # normalize
                x = x / (x.norm() + self.eps)

                # Newton-Schulz orthogonalization
                for _ in range(self.steps):
                    x = 1.5 * x - 0.5 * (x @ x.T @ x)

                if transpose:
                    x = x.T

                # convert back to original dtype
                x = x.to(p.dtype)

                # reshape
                x = x.reshape(orig_shape)

                # update
                p.add_(x, alpha=-lr)
