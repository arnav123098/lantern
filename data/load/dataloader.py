from abc import ABC
from lantern.data.prep.dataset_manager import datasets

'''
Dataloader base class
'''
class DataLoader(ABC):
    def __init__(self):
        super().__init__()

        self.state_keys = ()
        self.datasets = datasets()

    def state_dict(self):
        state_dict = {
            k: getattr(self, k)
            for k in self.state_keys
        }
        return state_dict

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            setattr(self, k, v)
