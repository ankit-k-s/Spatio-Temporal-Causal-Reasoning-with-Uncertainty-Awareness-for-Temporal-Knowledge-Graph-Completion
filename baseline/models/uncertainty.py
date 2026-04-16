import torch
import torch.nn as nn


class UncertaintyHead(nn.Module):
    def __init__(self, hidden_dim=128, num_entities=10000):
        super().__init__()

        self.mean_head = nn.Linear(hidden_dim, num_entities)
        self.log_var_head = nn.Linear(hidden_dim, num_entities)

    def forward(self, h):
        mu = self.mean_head(h)
        log_var = self.log_var_head(h)

        return mu, log_var