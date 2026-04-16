import torch
import torch.nn as nn


class GraphSSM(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()

        self.A = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.B = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, H_sequence):
        """
        H_sequence: list of tensors [H1, H2, ..., Ht]
        Each H: [num_nodes, hidden_dim]
        """

        S_prev = torch.zeros_like(H_sequence[0])
        S_all = []

        for H_t in H_sequence:
            S_t = self.A(S_prev) + self.B(H_t)
            S_all.append(S_t)
            S_prev = S_t

        return S_all  # list of temporal states