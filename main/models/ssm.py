import torch
import torch.nn as nn
from einops import einsum


class DiagonalSISOCell(nn.Module):
    def __init__(self, d_state, d_input):
        super().__init__()

        self.d_state = d_state

        # stable decay (HiPPO-inspired)
        self.A = nn.Parameter(-torch.exp(torch.linspace(0, 1, d_state)))
        self.B = nn.Parameter(torch.ones(d_input, d_state) * 0.1)
        self.C = nn.Parameter(torch.randn(d_input, d_state) * 0.1)

    def forward(self, x, state=None):
        N, D = x.shape

        if state is None:
            state = torch.zeros(N, D, self.d_state, device=x.device)

        A = self.A.view(1, 1, -1)

        Bx = einsum(x, self.B, 'n d, d s -> n d s')

        state = A * state + Bx
        state = torch.clamp(state, -50, 50)

        out = einsum(state, self.C, 'n d s, d s -> n d')

        return out, state