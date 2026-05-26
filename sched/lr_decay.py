from utils import dotdict
import math

'''
Learning rate (LR) drops slowly and smootly like cosine curve for a defined phase during learning (between warmup and max steps).
'''
def cosine_decay(
    config: dotdict,
    it: int
):
    # ramp up linearly
    if it < config.warmup_steps:
        lr = config.max_lr * (it + 1) / config.warmup_steps

    # if it's beyond max_steps, return min_lr
    if it > config.max_steps:
        lr = config.min_lr

    # cosine decay between warmup_steps and max_steps
    decay_ratio = (it - config.warmup_steps) / (config.max_steps - config.warnup_steps)
    decay_ratio = min(max(decay_ratio, 0), 1)
    assert 0 <= decay_ratio <= 1 # the line above holds this assertion

    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0

    lr = config.min_lr + coeff * (config.max_lr - config.min_lr)
    return lr
