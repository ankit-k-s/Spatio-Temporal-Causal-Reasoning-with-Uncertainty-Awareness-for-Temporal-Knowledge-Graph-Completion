import torch
import torch.nn as nn


class CausalModule(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, h):
        gamma = self.gate(h)
        h_causal = gamma * h
        return h_causal