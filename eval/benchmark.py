from abc import ABC, abstractmethod
from utils import dotdict

'''
For now this a very simple base class for benchmark evaluators.
'''
class Benchmark(ABC):
    def __init__(self, config: dotdict):
        self.config = config
        self.model = config.model

    @abstractmethod
    def evaluate(self): pass
