import torch
import torch.nn as nn
from einops import einsum, rearrange

class TemporalSSMLayer(nn.Module):
    """
    Stabilized continuous-time tracking layer.
    Incorporates LayerNorm, Gated Residuals, and reduced memory footprint.
    """
    def __init__(self, d_model: int, d_state: int = 16, use_mix: bool = False): # Fix 3: d_state = 16
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.use_mix = use_mix

        # Continuous-time parameters
        self.log_A = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.1)

        if self.use_mix:
            self.mix_proj = nn.Linear(d_model * 2, d_model)
            
        self.delta_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Softplus() 
        )

        # Fix 1 & 4: LayerNorm and Gating for SSM-on-SSM stability
        self.norm = nn.LayerNorm(d_model)
        self.W_g = nn.Linear(d_model, d_model)
        self.act = nn.SiLU()

    def forward(self, x_t: torch.Tensor, state: torch.Tensor, prev_x: torch.Tensor = None):
        if self.use_mix and prev_x is not None:
            mixed_rep = self.mix_proj(torch.cat([prev_x, x_t], dim=-1))
            delta = self.delta_proj(mixed_rep)
        else:
            delta = self.delta_proj(x_t)

        A_cont = -torch.exp(self.log_A)

        # Discretization
        A_disc = torch.exp(einsum(delta, A_cont, 'n d, d s -> n d s'))
        B_disc = einsum(delta, self.B, 'n d, d s -> n d s')

        # State Update
        new_state = A_disc * state + B_disc * rearrange(x_t, 'n d -> n d 1')

        # Output Projection
        out = einsum(new_state, self.C, 'n d s, d s -> n d')

        # Fix 1, 2 & 4: Gated Residual + LayerNorm
        gate = self.act(self.W_g(x_t))
        residual = x_t + gate * out
        
        return self.norm(residual), new_state


class GraphSSM(nn.Module):
    """
    CP-4: The full Temporal Encoder.
    """
    def __init__(self, d_model: int, d_state: int = 16, num_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            TemporalSSMLayer(d_model, d_state, use_mix=(i == 0))
            for i in range(num_layers)
        ])

    def forward(self, x_t: torch.Tensor, states: list = None, prev_xs: list = None):
        N, D = x_t.shape
        device = x_t.device

        if states is None:
            states = [torch.zeros((N, D, self.d_state), device=device) for _ in range(self.num_layers)]
        if prev_xs is None:
            prev_xs = [torch.zeros((N, D), device=device) for _ in range(self.num_layers)]

        new_states = []
        new_xs = []
        current_x = x_t

        for i, layer in enumerate(self.layers):
            out, new_state = layer(current_x, states[i], prev_xs[i])
            new_states.append(new_state)
            new_xs.append(current_x)  
            current_x = out

        return current_x, new_states, new_xs