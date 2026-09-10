import torch

class KVCache:
  def __init__(
      self,
      batch_size,
      n_head,
      block_size,
      head_size,
      n_layers,
      device,
      dtype
  ):
      self.batch_size = batch_size
      self.max_block_size = block_size
      self.n_layers = n_layers
      self.n_head = n_head
      self.head_size = head_size

      # Pre-allocate cache tensors: (n_layers, B, H, T, D)
      self.k_cache = torch.zeros(n_layers, batch_size, n_head, block_size, head_size, device=device, dtype=dtype)
      self.v_cache = torch.zeros(n_layers, batch_size, n_head, block_size, head_size, device=device, dtype=dtype)

      self.pos = 0

  def reset(self):
      self.pos = 0

  def get_layer_cache(self, layer_idx):
      return self.k_cache[layer_idx], self.v_cache[layer_idx]

  def advance(self, num_tokens):
      self.pos += num_tokens

  def store(self, layer_idx, k, v):
      # k, v: (B, n_head, T, head_size)

      T = k.size(2)

      if self.pos + T > self.max_block_size:
          raise ValueError("KV cache is full")

      self.k_cache[layer_idx, :, :, self.pos:self.pos + T] = k
      self.v_cache[layer_idx, :, :, self.pos:self.pos + T] = v

  def get(self, layer_idx):
      return (
          self.k_cache[
              layer_idx, :, :, :self.pos
          ],

          self.v_cache[
              layer_idx, :, :, :self.pos
          ]
      )
