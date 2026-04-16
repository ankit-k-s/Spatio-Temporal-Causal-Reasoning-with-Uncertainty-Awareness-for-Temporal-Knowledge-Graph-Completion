import torch
import torch.nn as nn

class NodeEmbedding(nn.Module):
    def __init__(self, num_entities, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(num_entities, hidden_dim)

    def forward(self):
        return self.embedding.weight  # [num_entities, hidden_dim]