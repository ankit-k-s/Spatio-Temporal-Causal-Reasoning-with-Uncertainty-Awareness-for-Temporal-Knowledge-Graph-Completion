import torch
import torch.nn as nn


# =========================
# Pure PyTorch scatter
# =========================
def scatter_mean(src, index, dim_size):
    out = torch.zeros(dim_size, src.size(1), device=src.device)
    count = torch.zeros(dim_size, device=src.device)

    out.index_add_(0, index, src)
    count.index_add_(0, index, torch.ones_like(index, dtype=torch.float))

    count = count.clamp(min=1).unsqueeze(1)
    return out / count


# =========================
# Graph Mamba Layer
# =========================
class GraphMambaLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        h = x.unsqueeze(0)  # (1, N, D)
        h, _ = self.attn(h, h, h)
        h = h.squeeze(0)

        x = x + h
        x = x + self.ffn(x)

        return self.norm(x)


# =========================
# Graph Mamba (CSI READY)
# =========================
class GraphMamba(nn.Module):
    def __init__(self, num_entities, num_relations, num_timestamps, dim, num_layers=2):
        super().__init__()

        self.entity_emb = nn.Embedding(num_entities, dim)
        self.rel_emb = nn.Embedding(num_relations, dim)
        self.time_emb = nn.Embedding(num_timestamps, dim)

        self.layers = nn.ModuleList([
            GraphMambaLayer(dim) for _ in range(num_layers)
        ])

    def forward(self, edge_index, edge_type, edge_time, edge_weight=None):
        src, dst = edge_index

        h_s = self.entity_emb(src)
        h_r = self.rel_emb(edge_type)
        h_t = self.time_emb(edge_time)

        # ------------------------
        # MESSAGE
        # ------------------------
        messages = h_s + h_r + h_t  # (E, D)

        # ------------------------
        # APPLY CSI MASK HERE 
        # ------------------------
        if edge_weight is None:
            edge_weight = torch.ones(messages.size(0), device=messages.device)

        messages = messages * edge_weight.unsqueeze(-1)

        # ------------------------
        # AGGREGATION
        # ------------------------
        num_nodes = self.entity_emb.num_embeddings
        x = scatter_mean(messages, dst, num_nodes)

        # ------------------------
        # MAMBA LAYERS
        # ------------------------
        for layer in self.layers:
            x = layer(x)

        return x