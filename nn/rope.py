import torch
import torch.nn as nn

'''
Modern LLMs use Rotary Positional Encodings (RoPE) which captures position of words relative to each other and works way better than absolute encodings. To achieve this, we just have to rotate the features by an angle that depends on the position of the token.

The original transformer paper proposed sinusoidal encodings for position where -

inverse frequency = 1/10000^(2i/n_dim)

[i is the index of the pair and n_dim is the head_size of the attn head]

RoPE makes use of this same formula to calculate the rotation angle.
We just multiply the token's position with the inverse frequency to get the rotation angle or theta.

Then we can get the cos and the sin using this theta and rotate the pairs to create rotary encodings.

This RoPE implementation is Llama style. (more coming soon)
'''
class RoPE(nn.Module):
    def __init__(self, head_size: int, rope_theta: float = 10000.0, llama_style: bool = False):
        assert head_size % 2 == 0 # works only on even head_size

        super().__init__()

        self.head_size = head_size # hs
        self.rope_theta = rope_theta

        self.llama_style = llama_style
    
    def get_theta(
        self,
        seq_len: int, # T
        device
    ):
        inv_freq = 1 / (
            self.rope_theta ** (torch.arange(0, self.head_size, 2, device=device).float() / self.head_size)
        ) # (hs // 2,)
        pos = torch.arange(seq_len, device=device) # (T,)

        theta = torch.outer(pos, inv_freq) # (T, hs // 2)
        return theta

    def apply_rope(
        self,
        x: torch.Tensor, # (B, nh, T, hs)
        sin: torch.Tensor,
        cos: torch.Tensor
    ):
        if self.llama_style:
            '''
            In the Llama style implementation, we split the features into two parts and then concat them as pairs of (-x2, x1).
            This looks like [-x3, -x4, x1, x2].
            Let's call this tensor y.
            Then we can element-wise multiply x by cos and y by sin and add them.
            This results in the same tensor as in the case of multiplying each pair by the rotation matrix.
            '''
            x1 = x[..., : self.head_size // 2] # [x3, x4] (B, nh, T, hs // 2)
            x2 = x[..., self.head_size // 2 :] # [x1, x2] (B, nh, T, hs // 2)
            
            rotated = torch.cat((-x2, x1), dim=-1) # [-x3, -x4, x1, x2] (B, nh, T, hs)
            rotated = x * cos + rotated * sin # (B, nh, T, hs)

            return rotated
        else:
            '''
            In other models like GPTNeoX, y looks like [-x2, x1, -x4, x3]. The math is identical. This difference in Llama is just because of lineage.
            '''
            x1 = x[..., 1::2]
            x2 = x[..., ::2]

            rotated = torch.stack((-x2, x1), dim=-1).flatten(-2) # stack creates a new dim so we need to flatten it back
            rotated = x * cos + rotated * sin

            return rotated

    def forward(self, x: torch.Tensor):
        device = x.device
        dtype = x.dtype

        seq_len = x.shape[-2]

        theta = self.get_theta(seq_len, device).to(dtype=dtype)
        sin, cos = theta.sin(), theta.cos() # (T, hs // 2)
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1) # (T, hs)

        return self.apply_rope(x, sin, cos) # (B, nh, T, hs)
