from lantern.utils import dotdict
import math

'''
Learning rate (LR) drops slowly and smootly like cosine curve for a defined phase during learning (between warmup and max steps).
'''
class CosineDecay:
    def __init__(self, max_lr: float, min_lr: float, max_steps: int, warmup_steps: int):
        assert max_lr > min_lr, "max_lr cannot be smaller than min_lr"
        assert max_steps > warmup_steps, "max_steps cannot be smaller than warmup_steps"

        self.max_lr = max_lr
        self.min_lr = min_lr
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        
    def __call__(self, it: int):
        # ramp up linearly
        if it < self.warmup_steps:
            lr = self.max_lr * (it + 1) / self.warmup_steps
            return lr

        # if it's beyond max_steps, return min_lr
        if it > self.max_steps:
            lr = self.min_lr
            return lr

        # cosine decay between warmup_steps and max_steps
        decay_ratio = (it - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        decay_ratio = min(max(decay_ratio, 0), 1)
        assert 0 <= decay_ratio <= 1 # the line above holds this assertion

        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0

        lr = self.min_lr + coeff * (self.max_lr - self.min_lr)
        return lr
