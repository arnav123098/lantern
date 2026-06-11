from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from nn.swiglu import FastSwiGLU
from nn.rope import RoPE
import math

# TODO: add documentation
# idea: make a wrapper for nn.Module to abstract out stuff like num_params, num_grad_params etc. and building blocks to build different models to make the whole build take way less lines and look cleaner

@dataclass
class TinyLlamaConfig:
    block_size: int = 2048
    vocab_size: int = 32000

    n_layer: int = 22

    n_head: int = 32
    n_kv_head: int = 4

    n_embd: int = 2048
    intermediate_size: int = 5632

    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5

class TinyLlama(nn.Module):
    def __init__(self, config: TinyLlamaConfig):
        super().__init__()
        self.config = config

        self.model = nn.ModuleDict(dict(
            embed_tokens = nn.Embedding(config.vocab_size, config.n_embd),
            layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            norm = nn.RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # tinyllama 1.1B doesn't share weights between wte and lm_head

        self.apply(self._init_weights)

    @property
    def num_params(self):
        return sum([p.numel() for p in self.parameters()])

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd))

        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
        elif isinstance(module, Attn):
            nn.init.normal_(
                module.c_proj.weight,
                mean=0.0,
                std=1 / math.sqrt(self.config.n_embd) / self.config.n_layer
            )

        elif isinstance(module, FastSwiGLU):
            nn.init.normal_(
                module.out_proj.weight,
                mean=0.0,
                std=1 / math.sqrt(self.config.n_embd) / self.config.n_layer
            )

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        T = idx.size(-1)
        assert T <= self.config.block_size, f"Cannot forward sequence of long length {T} while block size is only {self.config.block_size}"

        x = self.model.embed_tokens(idx)
        for h in self.model.layers:
            x = h(x)
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
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            context = x[:, -self.config.block_size:] # truncate if input tokens go beyond block_size
            logits, _ = self(context)
            logits = logits[:, -1, :] # we only need the last of T dim
            prob = F.softmax(logits, dim=-1) # calculate probabilities

            # use probabilities to get the next token and concat
            topk = min(topk, self.config.vocab_size)
            topk_probs, topk_idx = torch.topk(prob, topk, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_idx, -1, ix)
            x = torch.cat((x, xcol), dim=1)
        return x
    
    @classmethod
    def from_pretrained(cls, checkpoint: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"):
        from transformers import AutoModelForCausalLM

        print(f"loading weights from pretrained tinyllama 1.1B checkpoint: {checkpoint}")
        model = cls(TinyLlamaConfig)

        hf_model = AutoModelForCausalLM.from_pretrained(checkpoint)

        sd = model.state_dict()
        sd_keys = sd.keys()

        hf_sd = hf_model.state_dict()

        # copy
        for k in sd_keys:
            key = k
            if 'self_attn.attn' in k:
                base_key = k.replace('attn.weight', '')
                hf_attn = torch.cat([hf_sd[f"{base_key}{i}_proj.weight"] for i in ('q', 'k', 'v')], 0)
                
                assert hf_attn.shape == sd[k].shape

                with torch.no_grad():
                    sd[k].copy_(hf_attn)

                continue
            elif 'mlp.gatexvalue' in k:
                base_key = k.replace('gatexvalue.weight', '')
                hf_gatexvalue = torch.cat([hf_sd[f"{base_key}{i}_proj.weight"] for i in ('gate', 'up')], 0)

                assert hf_gatexvalue.shape == sd[k].shape

                with torch.no_grad():
                    sd[k].copy_(hf_gatexvalue)

                continue

            elif 'self_attn.c_proj' in k:
                key = k.replace('c_proj', 'o_proj')
            elif 'mlp.out_proj' in k:
                key = k.replace('out_proj', 'down_proj')

            assert hf_sd[key].shape == sd[k].shape

            with torch.no_grad():
                sd[k].copy_(hf_sd[key])

        print("weights loaded!")

        return model

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.post_attention_layernorm = nn.RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.self_attn = Attn(config)
        self.input_layernorm = nn.RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.mlp = FastSwiGLU(config.n_embd, config.intermediate_size, bias=False) # out = config.n_embd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class Attn(nn.Module): # GQA
    def __init__(
        self,
        config
    ):
        super().__init__()

        assert config.n_kv_head <= config.n_head
        assert config.n_head % config.n_kv_head == 0
        assert config.n_embd % config.n_head == 0

        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_size = config.n_embd // config.n_head
        
        self.kv_size = self.n_kv_head * self.head_size

        shape = (config.n_head + 2 * config.n_kv_head) * self.head_size

        self.attn = nn.Linear(config.n_embd, shape, bias=False)

        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.rope = RoPE(self.head_size, config.rope_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # create q, k, v
        qkv = self.attn(x)
        q, k, v = qkv.split((self.n_embd, self.kv_size, self.kv_size), dim=-1)

        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_size).transpose(1, 2)

        q, k = self.rope(q), self.rope(k)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out) # (B, T, C)
        return out

# model = TinyLlama(TinyLlamaConfig)
# model.half()
# print(model)
# print(model.num_params)
# print(sum(p.numel() for p in model.parameters() if p.requires_grad))

# x = torch.randint(0, 32000, (2, 128))

# logits, loss = model(x)

# print(logits.shape)
