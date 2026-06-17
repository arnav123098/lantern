import torch
import torch.nn as nn

'''
This file will contain common utilities for making stuff as well as for running experiments.
'''

class dotdict(dict):
    # dot.notation access to dictionary attributes
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def get_optim_groups(model: nn.Module, weight_decay: float | None = None):
    decay = []
    no_decay = []

    for p in model.parameters():
        if not p.requires_grad:
            continue

        if p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)

    decay_params = {"params": decay}
    if weight_decay is not None:
        decay_params["weight_decay"] = weight_decay

    return [
        decay_params,
        {"params": no_decay, "weight_decay": 0.0}
    ]

def load_weights(to_model: nn.Module, from_model: nn.Module, map: dict | None = None, grouped: dict | None = None, transposed: dict | None = None, filter: list | None = None):
  to_sd = to_model.state_dict()

  from_sd = from_model.state_dict()
  if filter is not None:
    to_sd = {k: v for k, v in to_sd.items() if not any(k.endswith(f) for f in filter)}
    from_sd = {k: v for k, v in from_sd.items() if not any(k.endswith(f) for f in filter)}

  to_sd_keys = to_sd.keys()

  for k in to_sd_keys:
    key = k
    loaded = False

    if grouped is not None:
      for tp, fp_list in grouped.items(): # fp -> from (model) pattern/layername, tp -> to (hf model) pattern/layername
        if k.endswith(f'{tp}.weight' if not (tp.endswith('.bias') or tp.endswith('.weight')) else tp):
          base_key = k.replace(f'{tp}.weight', '')
          from_weights = torch.cat([from_sd[f"{base_key}{i}.weight"] for i in fp_list], 0)

          assert from_weights.shape == to_sd[k].shape, f"Shape does not match for grouped weight {tp} <- {fp_list}"

          with torch.no_grad():
              to_sd[k].copy_(from_weights)

          loaded = True
          break
      if loaded: continue

    if transposed is not None:
      for tp, fp in transposed.items():
        if k.endswith(f'{tp}.weight'):
          key = k.replace(tp, fp)
          
          if k == 'transformer.h.0.attn.c_attn.bias':
            print('tp')

          assert from_sd[key].t().shape == to_sd[k].shape

          with torch.no_grad():
              to_sd[k].copy_(from_sd[key].t())

          loaded = True
          break
      if loaded: continue
   
    if map is not None:
      for tp, fp in map.items():
        if k.endswith(f'{tp}.weight' if not (tp.endswith('.bias') or tp.endswith('.weight')) else tp):
          key = k.replace(tp, fp)

          assert from_sd[key].shape == to_sd[k].shape

    assert key in from_sd, f"Missing key: {key}"

    with torch.no_grad():
      to_sd[k].copy_(from_sd[key])

  print('Weights loaded successfully!')

GPU_FLOPS = {
    "Tesla T4": {
        "fp32": 8.1e12,
        "fp16": 65e12
    },
    "A100-SXM4-40GB": {
        "fp32": 19.5e12,
        "tf32": 156e12,
        "fp16": 312e12,
        "bf16": 312e12,
    },
}

def estimate_mfu(model: nn.Module, tokens_per_sec: float | int | None = None, tokens_processed: float | int | None = None, dt: float | int | None = None, precision: str = 'fp32',gpu_name: str | None = None, peak_flops: float | int | None = None):
    assert gpu_name is not None or peak_flops is not None, "Any one of gpu_name and peak_flops is required"
    assert tokens_per_sec is not None or all([tokens_processed is not None, dt is not None])
    
    if peak_flops is None:
        assert gpu_name in GPU_FLOPS, f"No info about {gpu_name}'s promised flops"
        assert precision in GPU_FLOPS[gpu_name]
        
        peak_flops = GPU_FLOPS[gpu_name][precision]

    if tokens_per_sec is None:
        tokens_per_sec = tokens_processed / dt
        
    n_params = sum(p.numel() for p in model.parameters())
    estimated_flops = (6 * n_params) * tokens_per_sec
    mfu = estimated_flops / peak_flops
    return mfu
