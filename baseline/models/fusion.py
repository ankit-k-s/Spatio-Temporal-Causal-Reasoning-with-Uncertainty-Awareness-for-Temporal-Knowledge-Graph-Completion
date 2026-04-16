import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()

        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, H_t, S_t):
        """
        H_t: [num_nodes, hidden_dim]  (spatial)
        S_t: [num_nodes, hidden_dim]  (temporal)
        """

        combined = torch.cat([H_t, S_t], dim=1)

        gate = self.gate_net(combined)

        h_final = gate * H_t + (1 - gate) * S_t

        return h_final