import torch
from matplotlib import pyplot as plt
from collections.abc import Callable

class Metrics:
    def __init__(self):
        self.variables = {} # 'var_name': lambda: var_reference pairs (e.g. 'loss': lambda: loss)
        '''
        lambda is a safeguard in cases like model.wte.weight.norm() where we need to store this as a function and not a value returned by norm() once
        '''
        self.records = {} # 'var_name': [v1, v2, v3...]

    def track(self, variables: dict[str, Callable]): # register a metric once
        for k, fn in variables.items():
            if k in self.records:
                raise ValueError(f"Metric '{k}' already tracked")
            self.variables[k] = fn
            self.records[k] = []

    def record(self): # append new value in records
        for k, fn in self.variables.items():
            value = fn()

            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    value = value.item()
                else:
                    value = value.detach().cpu().clone()

            self.records[k].append(value)

    def get(self, metric: str):
        return self.records[metric]
    
    def tracked_metrics(self):
        return list(self.records.keys())

    def plot(self, metrics: str | list[str], title: str='metrics'):
        if isinstance(metrics, str):
            metrics = [metrics]
        
        for metric in metrics:
            record = self.get(metric)

            if isinstance(record, list):
                for i, r in record:
                    plt.plot(r, label=f'{metric}_{i}') 
            else:
                plt.plot(record, label=metric)

        plt.xlabel('steps')
        plt.legend()
        plt.title(title)
        plt.show()

    '''
    In order to save metrics and load them when resuming training, we need a state_dict.
    '''
    def state_dict(self):
        return {
            'records': self.records
        }
    
    def load_state_dict(self, state_dict: dict[str, dict]):
        self.records = state_dict['records']
