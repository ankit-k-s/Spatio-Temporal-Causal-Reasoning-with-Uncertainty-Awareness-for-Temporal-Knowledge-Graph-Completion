import torch
import torch.nn as nn


class GraphMamba(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()

        self.hidden_dim = hidden_dim

        # simple sequence model (like Mamba idea)
        self.linear_in = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, hidden_dim)

        self.activation = nn.ReLU()

    def forward(self, X, edge_index):
        """
        X: [num_nodes, hidden_dim]
        edge_index: [2, num_edges]
        """

        src = edge_index[0]
        dst = edge_index[1]

        # message passing
        messages = X[src]  # [num_edges, hidden_dim]

        # aggregate (mean)
        agg = torch.zeros_like(X)
        agg.index_add_(0, dst, messages)

        # normalize
        deg = torch.zeros(X.size(0), device=X.device)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))

        deg = deg.clamp(min=1).unsqueeze(1)
        agg = agg / deg

        # Mamba-style transformation
        h = self.linear_in(agg)
        g = torch.sigmoid(self.gate(agg))

        out = self.linear_out(self.activation(h) * g)

        return out