import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Swish Gated Linear Unit or SwiGLU is made up of two things - a Swish Activation and a Gated Linear Unit (GLU).

Swish: x * sigmoid(beta * x) where beta is a scalar (usually 1) and x is the input
GLU: sigmoid(W1 * x + b1) * (W2 * x + b2)

In simple terms, SwiGLU helps decide how much of information we want to keep i.e. if we want to amplify the signal or make it less powerful.
This is implemented as follows:
- gate = linear1(x), value = linear2(x)
- now calculate y = Swish(gate) * value <- this is what SwiGLU essentially does
- to get results in model's hidden dim, use linear(y) as out_proj
'''
class SwiGLU(nn.Module):
    def __init__(
        self,
        fan_in: int,
        fan_out: int,
        bias: bool = False
    ):
        super().__init__()

        self.gate_proj = nn.Linear(fan_in, fan_out, bias)
        self.value_proj = nn.Linear(fan_in, fan_out, bias)
        self.out_proj = nn.Linear(fan_out, fan_in, bias)

    def forward(self, x: torch.Tensor):
        x = F.silu(self.gate_proj(x)) * self.value_proj(x) # SILU is the Swish function
        return self.out_proj(x)

'''
To make the operations faster, we use just one linear layer to calculate both gate and value which requires just one GEMM instead of two.
'''
class FastSwiGLU(nn.Module):
    def __init__(
        self,
        fan_in: int,
        fan_out: int,
        bias: bool = False
    ):
        super().__init__()

        self.fc = nn.Linear(fan_in, 2 * fan_out, bias)
        self.out_proj = nn.Linear(fan_out, fan_in, bias)

    def forward(self, x: torch.Tensor):
        gate, value = self.fc(x).chunk(2, dim=-1)
        x = F.silu(gate) * value
        return self.out_proj(x)
