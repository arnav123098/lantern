# DEEPSEEK V2 (a kinda generic one)

import math
from dataclasses import dataclass
from nn.models.model import Model
from nn.attn import RoPE
from nn.mlp import FastSwiGLU

# not enough comments/documentation for now, but i'll write an article soon

@dataclass
class DeepseekV2Config: # this config is literally anything (totes random). please don't copy it.
    block_size: int = 2048
    vocab_size: int = 1024

    n_layer: int = 5

    n_head: int = 8
    n_embd: int = 216

    n_routed_experts: int = 4
    expert_topk: int = 1
    n_shared_experts: int = 1
    d_expert: int = 128

    c_dim: int = 64
    r_dim: int = 16

    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6

class DeepseekV2(Model):
    def __init__(self, config: DeepseekV2Config):
        super().__init__()
        self.config = config

        self.model = nn.ModuleDict(dict(
            embed_tokens = nn.Embedding(config.vocab_size, config.n_embd),
            layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            norm = nn.RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    @property
    def num_params(self):
        return sum([p.numel() for p in self.parameters()])

    def forward(
        self,
        idx: torch.Tensor,
        cache = None,
        targets: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        T = idx.size(-1)
        assert T <= self.config.block_size, f"Cannot forward sequence of long length {T} while block size is only {self.config.block_size}"

        x = self.model.embed_tokens(idx)

        for i, h in enumerate(self.model.layers):
            x = h(x, cache=cache, layer_idx=i)

        x = self.model.norm(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor, # x is B x T
        max_new_tokens: int = 500,
        topk: int = 50
    ) -> torch.Tensor: # without cache
        for _ in range(max_new_tokens):
            context = x[:, -self.config.block_size:] # truncate if input tokens go beyond block_size
            logits, _ = self(context)
            logits = logits[:, -1, :]
            prob = F.softmax(logits, dim=-1)

            topk = min(topk, self.config.vocab_size)
            topk_probs, topk_idx = torch.topk(prob, topk, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_idx, -1, ix)
            x = torch.cat((x, xcol), dim=1)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.post_attention_layernorm = nn.RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.self_attn = DeepseekMLA(config, rope=RoPE(config.r_dim))
        self.input_layernorm = nn.RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.mlp = MoE(config.n_embd, config.d_expert, config.n_routed_experts, config.n_shared_experts, config.expert_topk)

    def forward(self, x: torch.Tensor, cache = None, layer_idx = None) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cache=cache, layer_idx=layer_idx)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class DeepseekMLA(nn.Module):
  def __init__(
      self,
      config,
      rope = None
  ):
      super().__init__()

      self.n_embd = config.n_embd

      self.n_head = config.n_head
      self.head_size = config.n_embd // config.n_head

      self.c_dim = config.c_dim
      self.r_dim = config.r_dim

      self.q_down = nn.Linear(config.n_embd, self.c_dim, bias=False)
      self.q_up = nn.Linear(self.c_dim, config.n_embd, bias=False)
      self.q_r = nn.Linear(self.c_dim, self.n_head * self.r_dim, bias=False)

      self.kv_down = nn.Linear(config.n_embd, self.c_dim, bias=False)
      self.k_up = nn.Linear(self.c_dim, config.n_embd, bias=False)
      self.v_up = nn.Linear(self.c_dim, config.n_embd, bias=False)
      self.k_r = nn.Linear(self.c_dim, self.r_dim, bias=False)

      self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

      self.rope = rope

  def forward(self, x: torch.Tensor, cache = None, layer_idx = None):
      B, T, _ = x.shape # B, T, C

      qd = self.q_down(x) # B, T, c_dim
      qu = self.q_up(qd).view(B, T, self.n_head, self.head_size)
      qr = self.q_r(qd).view(B, T, self.n_head, self.r_dim).transpose(1, 2) # B, nh, T, rs
      qr = self.rope.forward(qr, start_pos=(cache.pos if cache is not None else 0)) # B, nh, T, rs

      kvd = self.kv_down(x) # B, T, c_dim

      kr = self.k_r(kvd) # B, T, rs
      kr = self.rope.forward(kr.reshape(B, 1, T, self.r_dim), start_pos=(cache.pos if cache is not None else 0)).reshape(B, T, self.r_dim)

      if cache is not None:
          cache.store(layer_idx, kvd, kr)
          kvd, kr = cache.get(layer_idx, end_pos=T)

      ku = self.k_up.weight.view(self.n_head, self.head_size, self.c_dim)

      # qu @ ku
      q_absorbed = torch.einsum(
          'bthd,hdc->bthc',
          qu,
          ku
      )

      # content attn
      scores_c = torch.einsum(
          'bthc,bsc->bhts',
          q_absorbed,
          kvd
      )

      # rope attn
      scores_r = torch.einsum(
          'bhtr,bsr->bhts',
          qr,
          kr
      )

      scores = (scores_c + scores_r) / (self.head_size + self.r_dim) ** 0.5
      attn = torch.softmax(scores, dim=-1)

      context = torch.einsum(
          'bhts,bsc->bhtc',
          attn,
          kvd
      )

      vu = self.v_up.weight.view(self.n_head, self.head_size, self.c_dim)

      context = torch.einsum(
          'bhtc,hdc->bthd',
          context,
          vu
      ).reshape(B, T, self.n_embd)

      return self.o_proj(context)

class MoE(nn.Module):
  def __init__(
      self,
      d_model,
      d_expert,
      n_routed_experts,
      n_shared_experts,
      topk
  ):
      super().__init__()

      self.n_routed_experts = n_routed_experts
      self.n_shared_experts = n_shared_experts
      self.topk = topk

      self.router = nn.Linear(d_model, n_routed_experts, bias=False)

      self.shared_experts = nn.ModuleList([
          FastSwiGLU(d_model, d_expert) # d_model -> d_expert (intermediate) -> d_model
          for _ in range(n_shared_experts)
      ])
      self.experts = nn.ModuleList([
          FastSwiGLU(d_model, d_expert)
          for _ in range(n_routed_experts)
      ])

  def forward(self, x):
      logits = self.router(x) # B, T, n_routed_experts
      probs = F.softmax(logits, dim=-1)

      topk_weights, topk_indices = torch.topk(
          probs,
          self.topk,
          dim=-1
      ) # both are B, T, topk

      # norm
      topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

      shared_out = torch.zeros_like(x) # B, T, d_model
      for exp in self.shared_experts:
          shared_out += exp(x)

      routed_out = torch.zeros_like(x)
      for i, exp in enumerate(self.experts):
          mask = topk_indices == i # B, T, topk

          if not mask.any(): continue

          # suppose N tokens in total choose this expert

          B, T, topk = mask.nonzero(as_tuple=True) # N
          weight = topk_weights[B, T, topk] # N

          routed_out[B, T] += weight[:, None] * exp(x[B, T]) # N, C

      return shared_out + routed_out # B, T, d_model

# model = DeepseekV2(DeepseekV2Config)
# print(model)
# print(model.num_params)
