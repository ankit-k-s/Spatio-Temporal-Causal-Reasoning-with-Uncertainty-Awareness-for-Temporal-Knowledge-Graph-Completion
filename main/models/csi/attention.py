import torch
import torch.nn as nn

class CSIAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.e = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, 1)

    def forward(self, query, edge_embs):
        q = self.q(query).unsqueeze(1)
        e = self.e(edge_embs).unsqueeze(0)

        scores = torch.tanh(q + e)
        scores = self.out(scores).squeeze(-1)

        return torch.sigmoid(scores)