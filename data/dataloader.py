from abc import ABC
from downloader import Downloader

'''
Dataloader base class
'''
class DataLoader(ABC):
    def __init__(self):
        super().__init__()

        self.state_keys = ()
        self.downloader = Downloader()

    def state_dict(self):
        state_dict = {
            k: getattr(self, k)
            for k in self.state_keys
        }
        return state_dict

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            setattr(self, k, v)
