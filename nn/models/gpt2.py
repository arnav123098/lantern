from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from lantern.nn.attn import MultiHeadAttn

@dataclass
class GPT2Config:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

'''
GPT in a nutshell:
- take information about position of input i.e. wpe(input); and it's token embedding wte(input); combine them (let's call it x)
- apply attn mechanism on x and then take it through an mlp (this is a block)
- repeat n_layer times
- apply one last linear transformation to get B x T x vocab_size tensor as logits
- now the last dim can be softmax-ed to form probabilites
'''
class GPT2(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # lm_head and wte share weights
        self.lm_head.weight = self.transformer.wte.weight

        self.apply(self._init_weights) # initialize weights as defined by the function below

    '''
    When replicating a model, it's better to get info about what stuff the original one used for initialization and then write _init_weigths in accordance with it.
    '''
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            std = 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) # initialize biases as 0
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def num_params(self):
        return sum([p.numel() for p in self.parameters()])

    def forward(
        self,
        idx: torch.Tensor, # idx is B x T
        targets: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        T = idx.size(-1)
        assert T <= self.config.block_size, f"Cannot forward sequence of long length {T} while block size is only {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = pos_emb + tok_emb

        for h in self.transformer.h:
            x = h(x, attn_mask)
        x = self.transformer.ln_f(x)
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
    
    '''
    This from_pretrained method is used to load weights from a pretrained gpt2 from huggingface. This implementation closely follows Karpathy's implementation in NanoGPT.

    PS: You can skip this if you don't wanna get so much hands on with weight loading. For easy weight loading, I've made a utility in Lantern.
    '''
    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

        override_args = override_args or {} 
        assert all(k == 'dropout' for k in override_args) # allow only dropout to be overridden

        from transformers import GPT2LMHeadModel
        print(f"loading weights from pretrained gpt: {model_type}")

        # set config
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]

        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True

        if 'dropout' in override_args: # override dropout
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']

        # make model
        config = GPT2Config(**config_args)
        model = cls(config)

        # get state_dict
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')] # remove mask from params

        # load model from huggingface and get state_dict
        hf_model = GPT2LMHeadModel.from_pretrained(model_type)
        hf_sd = hf_model.state_dict()
        hf_sd_keys = [k for k in hf_sd.keys() if not (k.endswith('.attn.bias') or k.endswith('.attn.masked_bias'))]

        # OpenAI checkpoints use "Conv1D" module.
        # Pytorch stores w as (out_features, in_features) and does x @ w.T but GPT2's Conv1D does x @ w so we need to transpose these weights before copying them.
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(hf_sd_keys) == len(sd_keys), f"mismatched keys: {len(hf_sd_keys)} != {len(sd_keys)}"

        # copying params
        for k in hf_sd_keys:
            if any(k.endswith(w) for w in transposed):
                # weights to transpose
                assert hf_sd[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(hf_sd[k].t())
            else:
                # vanilla copy
                assert hf_sd[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(hf_sd[k])

        return model
    
    '''
    # same as Andrej's configure_optimizers method in nanogpt 

    PS: you can skip this as its not a part of the model but smth that'll be used in training especially with the BasicTrainer class and so i made it
    '''
    def configure_optimizers(self, weight_decay, learning_rate, **kwargs):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        from lantern.optim.adam import AdamW
        optimizer = AdamW(optim_groups, lr=learning_rate, weight_decay=weight_decay)

        return optimizer

'''
GPT has many units or blocks which are just attn followed by mlp/feed-forward.
'''
class Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttn(config, is_causal=True)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attn_mask) # we are using pre-layer-norm here
        x = x + self.mlp(self.ln_2(x))
        return x
    
'''
A straightforward and simple feed-forward layer that follows attn
'''
class MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
