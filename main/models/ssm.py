import torch
import torch.nn as nn
from einops import einsum


class DiagonalSISOCell(nn.Module):
    def __init__(self, d_state, d_input):
        super().__init__()

        self.d_state = d_state
        self.d_input = d_input

        self.log_nA = nn.Parameter(torch.randn(d_input, d_state))
        self.B = nn.Parameter(torch.ones(d_input, d_state))
        self.C = nn.Parameter(torch.randn(d_input, d_state, d_state))

        self.delta = nn.Sequential(
            nn.Linear(d_input, 1),
            nn.Softplus()
        )

    def forward(self, x, state=None):
        N, D = x.shape

        if state is None:
            state = torch.zeros(N, D, self.d_state, device=x.device)

        delta = self.delta(x).squeeze(-1)

        A = -torch.exp(self.log_nA)

        A_zoh = torch.exp(einsum(delta, A, 'n, d s -> n d s'))
        B = einsum(delta, self.B, 'n, d s -> n d s')
        Bx = einsum(B, x, 'n d s, n d -> n d s')

        state = A_zoh * state + Bx

        out = einsum(state, self.C, 'n d s, d s t -> n d t')

        return out