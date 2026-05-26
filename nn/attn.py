import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import dotdict

# SelfAttn and MultiHeadAttn are GPT 2 and 3 style
'''
Self attention mechanism allow tokens to talk to each other using query (q), key (k) and value (v) vectors. Some q-k dot products produce higher scores indicating a significant relation between such tokens.

- a token creates q, k, v vectors
- the q vector searches for k vectors from other tokens
- if causal, this search is limited to tokens upto the current one i.e. future tokens cannot be attented
- the dot products produce a weight matrix
- weights @ v produces the final attn scores
'''
class SelfAttn(nn.Module):
    def __init__(
        self,
        config: dotdict,
        is_causal: bool = False
    ):
        super().__init__()

        self.query = nn.Linear(config.n_embd, config.head_size, bias=False)
        self.key = nn.Linear(config.n_embd, config.head_size, bias=False)
        self.value = nn.Linear(config.n_embd, config.head_size, bias=False)

        self.n_embd = config.n_embd
        self.head_size = config.head_size

        self.is_causal = is_causal

    def forward(self, x: torch.Tensor) -> torch.Tensor: # x is a B x T x C tensor
        T = x.size(-2)
        q, k, v = self.query(x), self.key(x), self.value(x) # (B, T, head_size)

        # raw attn scores
        weights = q @ k.transpose(-2, -1)

        # in causal self attn, tokens do not attend to future tokens, so add mask
        if self.is_causal:
            mask = torch.tril(torch.ones(T, T), device=x.device, dtype=torch.bool)
            weights = weights.masked_fill(mask == 0, float('-inf'))
        
        scaling_factor = self.head_size ** -0.5 # for training stability

        # normalize
        weights = F.softmax(weights * scaling_factor, dim=-1)
        
        out = weights @ v # (B, T, head_size)
        return out
    
'''
The same self attention mechanism but now there are many self_attn heads combined. We just use concatenated qkv, and then split them and reshape for n_head number of heads before computing attn_scores.
'''
class MultiHeadAttn(nn.Module):
    def __init__(
        self,
        config: dotdict,
        is_causal: bool = False
    ):
        super().__init__()

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_embd = config.n_embd
        self.n_head = config.n_head # number of attn heads

        '''
        self.register_buffer('bias', torch.tril(torch.zeros(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

        *We are not using bias/mask. There's a better way to do this in pytorch using F.scaled_dot_product_attention.
        '''

        self.is_causal = is_causal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        head_size = C // self.n_head

        # create qkv
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)

        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)

        # raw attn scores
        '''
        don't need this:

        weights = q @ k.transpose(-1, -2)
        scaling_factor = head_size ** -0.5

        if is_causal:
            weights = weights.masked_fill(self.bias[:, :, T, T] == 0, float('-inf'))
            weights = F.softmax(weights * scaling_factor, dim=-1)

        out = weights @ v

        instead there's a better way -'''

        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal) # (B, n_head, T, T)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out) # (B, T, C)
        return out
