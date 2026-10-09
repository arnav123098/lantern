# DEEPSEEK V2 (replicating shibatch deepseek v2 3m)
import math
from dataclasses import dataclass
from nn.models.model import Model
from nn.attn import RoPE
from nn.mlp import FastSwiGLU
from utils import load_weights

"""
A few things to note:
param_count: 2928392

this model's logits vs shibatch deepseek v2 3m logits (on a random input):
max: tensor(36.0653, grad_fn=<MaxBackward1>)
mean: tensor(1.7380, grad_fn=<MeanBackward0>)
rmse: tensor(2.2962, grad_fn=<SqrtBackward0>)
cos: tensor(0.9338, grad_fn=<SumBackward1>)

So, not perfect, but close enough.
"""

@dataclass
class DeepseekV2Config:
    block_size: int = 2048
    vocab_size: int = 1024

    n_layer: int = 5

    n_head: int = 8
    n_embd: int = 216

    qk_nope_head_dim: int = 16
    qk_rope_head_dim: int = 16
    v_head_dim: int = 32
    kv_lora_rank: int = 64

    block0_intermediate: int = 432

    n_routed_experts: int = 4
    expert_topk: int = 1
    n_shared_experts: int = 1
    d_expert: int = 128 # swiglu intermediate, to be precise

    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6

class DeepseekV2(Model):
    def __init__(self, config: DeepseekV2Config):
        super().__init__()
        self.config = config

        self.model = nn.ModuleDict(dict(
            embed_tokens = nn.Embedding(config.vocab_size, config.n_embd),
            layers = nn.ModuleList([Block0(config)] + [Block(config) for _ in range(config.n_layer - 1)]),
            norm = nn.RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # shibatch v2 3m has tied embeddings so to match them -
        self.lm_head.weight = self.model.embed_tokens.weight

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

    @classmethod
    def from_pretrained(cls, to_model, from_model):
        """
        Btw this is how you load the Shibata-san's model:

        from pathlib import Path
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer

        repo = "shibatch/tinydeepseekv2-3m"
        model_dir = Path(snapshot_download(
            repo_id=repo,
            allow_patterns=["hf/*"],
        )) / "hf"

        from_model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            dtype=torch.float32,
        )
        """

        to_sd = to_model.state_dict()
        from_sd = from_model.state_dict()

        to_sd_keys = to_sd.keys()
        from_sd_keys = from_sd.keys()

        keys = []
        for k in from_sd_keys:
            if '.experts' in k:
                key = '.'.join(k.split('.')[:-1])
                if key not in keys: keys.append(key)

        for i, k in enumerate(keys):
            g, d = from_sd[f'{k}.gate_up_proj'][i], from_sd[f'{k}.down_proj'][i]
            keyg, keyd = k + f'.{i}.gatexvalue.weight', k + f'.{i}.out_proj.weight'

            with torch.no_grad():
            to_sd[keyg].copy_(g)
            to_sd[keyd].copy_(d)

        filter = []
        for i in range(to_model.config.n_routed_experts):
            filter.extend([f'experts.{i}.gatexvalue.weight', f'experts.{i}.out_proj.weight'])

        load_weights(to_model, from_model, map={
            'mlp0.out_proj': 'mlp.down_proj',
            'shared_experts.out_proj': 'shared_experts.down_proj',
            'router': 'gate'
        },
        grouped={
            '0.mlp0.gatexvalue': ['0.mlp.gate_proj', '0.mlp.up_proj'],
            'shared_experts.gatexvalue': ['shared_experts.gate_proj', 'shared_experts.up_proj']
        },
        filter=filter
        )

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.post_attention_layernorm = nn.RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.self_attn = DeepseekMLA(config, rope=RoPE(config.qk_rope_head_dim))
        self.input_layernorm = nn.RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.mlp = MoE(config.n_embd, config.d_expert, config.n_routed_experts, config.n_shared_experts, config.expert_topk)

    def forward(self, x: torch.Tensor, cache = None, layer_idx = None) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cache=cache, layer_idx=layer_idx)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class Block0(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.post_attention_layernorm = nn.RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.self_attn = DeepseekMLA(config, rope=RoPE(config.qk_rope_head_dim))
        self.input_layernorm = nn.RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.mlp0 = FastSwiGLU(config.n_embd, config.block0_intermediate, bias=False)

    def forward(self, x: torch.Tensor, cache = None, layer_idx = None) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cache=cache, layer_idx=layer_idx)
        x = x + self.mlp0(self.post_attention_layernorm(x))
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

      self.qk_nope_head_dim = config.qk_nope_head_dim
      self.qk_rope_head_dim = config.qk_rope_head_dim
      self.v_head_dim = config.v_head_dim
      self.kv_lora_rank = config.kv_lora_rank

      self.q_proj = nn.Linear(
          config.n_embd,
          self.n_head * (self.qk_nope_head_dim + self.qk_rope_head_dim),
          bias=False,
      )

      self.kv_a_proj_with_mqa = nn.Linear(
          config.n_embd,
          self.kv_lora_rank + self.qk_rope_head_dim,
          bias=False,
      )

      self.kv_a_layernorm = nn.RMSNorm(
          self.kv_lora_rank,
          eps=config.rms_norm_eps,
      )

      self.kv_b_proj = nn.Linear(
          self.kv_lora_rank,
          self.n_head * (self.qk_nope_head_dim + self.v_head_dim),
          bias=False,
      )

      self.o_proj = nn.Linear(
          self.n_head * (self.qk_nope_head_dim + self.qk_rope_head_dim),
          config.n_embd,
          bias=False,
      )

      self.rope = rope

  def forward(self, x: torch.Tensor, cache = None, layer_idx = None):
      B, T, _ = x.shape # B, T, C

      q = self.q_proj(x).view(B, T, self.n_head, self.qk_nope_head_dim + self.qk_rope_head_dim) # B, T, H, qk_nope_head_dim + R
      q_nope, q_rope = q[..., :self.qk_nope_head_dim], q[..., self.qk_nope_head_dim:]

      q_rope = q_rope.transpose(1, 2) # B, H, T, R
      q_rope = self.rope(
          q_rope,
          start_pos=(cache.pos if cache is not None else 0)
      )

      kv = self.kv_a_proj_with_mqa(x) # B, T, C + R
      kv_latent, k_rope = kv[..., :self.kv_lora_rank], kv[..., self.kv_lora_rank:]

      kv_latent = self.kv_a_layernorm(kv_latent) # normalizing because shibatch does too

      k_rope = self.rope(
          k_rope.unsqueeze(1),
          start_pos=(cache.pos if cache is not None else 0)
      ).squeeze(1) # B, T, R

      if cache is not None:
          cache.store(layer_idx, kv_latent, k_rope)
          kv_latent, k_rope = cache.get(layer_idx, end_pos=T) # B, S, C; B, S, R

      kv_b = self.kv_b_proj(kv_latent).view(
          B, -1,
          self.n_head,
          self.qk_nope_head_dim + self.v_head_dim
      )

      k_nope, v = kv_b[..., :self.qk_nope_head_dim], kv_b[..., self.qk_nope_head_dim:] # both B, S, H, D (D = qk_nope_head_dim)

      scores_nope = torch.einsum(
          'bthd,bshd->bhts', # t is the current seq_len i.e. 1 if cache; s is the total seq_len used
          q_nope,
          k_nope
      )
      scores_r = torch.einsum(
          'bhtr,bsr->bhts',
          q_rope,
          k_rope
      )
      scores = (scores_nope + scores_r) / (self.qk_nope_head_dim + self.qk_rope_head_dim) ** 0.5
      attn = torch.softmax(scores, dim=-1)

      context = torch.einsum(
          'bhts,bshv->bthv',
          attn,
          v
      ).reshape(B, T, self.n_head * self.v_head_dim)

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
          FastSwiGLU(d_model, d_expert, bias=False) # d_model -> d_expert (intermediate) -> d_model
          for _ in range(n_shared_experts)
      ])
      if n_shared_experts == 1: # have to do this to load weights correctly as it matches the way shibatch has it
          self.shared_experts = self.shared_experts[0]

      self.experts = nn.ModuleList([
          FastSwiGLU(d_model, d_expert, bias=False)
          for _ in range(n_routed_experts)
      ])
      if n_routed_experts == 1:
          self.experts = self.experts[0]

  def forward(self, x):
      if not type(self.shared_experts) == list: shared_experts = [self.shared_experts]
      if not type(self.experts) == list: experts = [self.experts]

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
      for exp in shared_experts:
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
