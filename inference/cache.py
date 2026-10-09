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

  def get(self, layer_idx, T=None):
      k_cache, v_cache = (
          self.k_cache[
              layer_idx, :, :, :
          ],

          self.v_cache[
              layer_idx, :, :, :
          ]
      )

      if T is not None:
          k_cache, v_cache = k_cache[:, :self.pos+T, :], v_cache[:, :self.pos+T, :]

      return k_cache, v_cache

class MLACache:
  def __init__(
      self,
      batch_size,
      block_size,
      c_dim,
      r_dim,
      n_layers,
      device,
      dtype
  ):
      self.batch_size = batch_size
      self.max_block_size = block_size
      self.n_layers = n_layers
      self.c_dim = c_dim
      self.r_dim = r_dim

      # Pre-allocate cache tensors: (n_layers, B, T, D)
      self.kv_latent = torch.zeros(n_layers, batch_size, block_size, c_dim, device=device, dtype=dtype)
      self.k_rope = torch.zeros(n_layers, batch_size, block_size, r_dim, device=device, dtype=dtype)

      self.pos = 0

  def reset(self):
      self.pos = 0

  def get_layer_cache(self, layer_idx):
      return self.kv_latent[layer_idx], self.k_rope[layer_idx]

  def advance(self, num_tokens):
      self.pos += num_tokens

  def store(self, layer_idx, kv_latent, k_rope):
      # kv_latent, k_rope: (B, T, D)

      T = kv_latent.size(1)

      if self.pos + T > self.max_block_size:
          raise ValueError("MLA cache is full")

      self.kv_latent[layer_idx, :, self.pos:self.pos + T] = kv_latent
      self.k_rope[layer_idx, :, self.pos:self.pos + T] = k_rope

  def get(self, layer_idx, T=None):
      kvd, kr = (
          self.kv_latent[
              layer_idx, :, :
          ],
          self.k_rope[
              layer_idx, :, :
          ]
      )

      if T is not None:
          kvd, kr = kvd[:, :self.pos+T, :], kr[:, :self.pos+T, :]

      return kvd, kr