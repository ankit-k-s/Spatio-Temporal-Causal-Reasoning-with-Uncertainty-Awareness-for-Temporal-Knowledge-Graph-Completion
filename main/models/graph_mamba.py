import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# SCATTER MEAN
# =========================
def scatter_mean(src, index, dim_size, weight=None):
    out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    count = torch.zeros(dim_size, device=src.device, dtype=src.dtype)

    if weight is not None:
        weight = weight.to(src.dtype)
        src = src * weight.unsqueeze(-1)

    out.index_add_(0, index, src)

    if weight is None:
        count.index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    else:
        count.index_add_(0, index, weight)

    count = count.clamp(min=1).unsqueeze(1)
    return out / count


# =========================
# MAMBA LAYER (SAFE, NO INPLACE)
# =========================
class MambaLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.in_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)

        self.A = nn.Parameter(torch.randn(dim) * 0.1)
        self.B = nn.Parameter(torch.randn(dim) * 0.1)

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (N, D)

        z, g = self.in_proj(x).chunk(2, dim=-1)
        g = torch.sigmoid(g)

        # SAFE recurrence (NO inplace ops)
        h_list = []
        prev = torch.zeros_like(z[0])

        for i in range(z.size(0)):
            curr = g[i] * (self.A * prev) + self.B * z[i]
            h_list.append(curr)
            prev = curr

        h = torch.stack(h_list, dim=0)

        out = self.out_proj(h)

        return self.norm(x + out)


# =========================
# GRAPH MAMBA
# =========================
class GraphMamba(nn.Module):
    def __init__(self, num_entities, num_relations, num_timestamps, dim, num_layers=2):
        super().__init__()

        self.num_entities = num_entities

        self.entity_emb = nn.Embedding(num_entities, dim)
        self.rel_emb = nn.Embedding(num_relations, dim)
        self.time_emb = nn.Embedding(num_timestamps, dim)

        self.layers = nn.ModuleList([
            MambaLayer(dim) for _ in range(num_layers)
        ])

    def forward(self, edge_index, edge_type, edge_time, edge_weight=None):
        src, dst = edge_index

        # =========================
        # EMBEDDINGS
        # =========================
        h_s = self.entity_emb(src)
        h_r = self.rel_emb(edge_type)
        h_t = self.time_emb(edge_time)

        # =========================
        # MESSAGE PASSING
        # =========================
        messages = h_s + h_r + h_t

        x = scatter_mean(messages, dst, self.num_entities, weight=edge_weight)

        # normalize for stability
        x = F.normalize(x, dim=1)

        # pseudo ordering (important for sequence modeling)
        perm = torch.randperm(x.size(0), device=x.device)
        x = x[perm]

        # =========================
        # MAMBA LAYERS
        # =========================
        for layer in self.layers:
            x = layer(x)

        return x