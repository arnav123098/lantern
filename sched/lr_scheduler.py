from abc import ABC, abstractmethod

class LRScheduler(ABC):
    @abstractmethod
    def get_lr(self, step: int)  -> float: pass
