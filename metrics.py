import torch
from matplotlib import pyplot as plt
from collections.abc import Callable

import psutil
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates

class Metrics:
    def __init__(self):
        self.variables = {} # 'var_name': lambda: var_reference pairs (e.g. 'loss': lambda: loss)
        '''
        lambda is a safeguard in cases like model.wte.weight.norm() where we need to store this as a function and not a value returned by norm() once
        '''
        self.records = {} # 'var_name': [v1, v2, v3...]

        self.sys = None

    def create_sys_metrics(self):
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        util = nvmlDeviceGetUtilizationRates(handle)

        self.sys = {
            "gpu_util": lambda: util.gpu,
            "mem_util": lambda: util.memory,

            "cpu_percent": lambda: psutil.cpu_percent(),
            "ram_used": lambda: round(psutil.virtual_memory().used / 1024**3, 2),

            "gpu_mem_alloc": lambda: round(torch.cuda.memory_allocated() / 1024**3, 2),
            "gpu_mem_reserved": lambda: round(torch.cuda.memory_reserved() / 1024**3, 2),
            "peak_gpu_mem": lambda: round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        }

    def track(self, variables: dict[str, Callable]): # register a metric once
        for k, fn in variables.items():
            if k in self.records:
                print(f"Already tracking metric '{k}'")
                continue
            
            if k == 'sys':
                self.create_sys_metrics()
                print("Monitoring system-related metrics")

                for sys_k, sys_fn in self.sys.items():
                    self.variables[sys_k] = sys_fn
                    self.records[sys_k] = []
                continue

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

    def plot(self, metrics: str | list[str], title: str='metrics'): # super-basic plotting
        if isinstance(metrics, str):
            metrics = [metrics]
        
        for metric in metrics:
            record = self.get(metric)

            if all(isinstance(x, list) for x in record):
                for i, r in enumerate(record):
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
