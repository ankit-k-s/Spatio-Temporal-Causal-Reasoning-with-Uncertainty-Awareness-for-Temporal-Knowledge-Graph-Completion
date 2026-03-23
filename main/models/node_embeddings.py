import torch
import torch.nn as nn

class NodeEmbedding(nn.Module):
    def __init__(self, num_entities, dim):
        super().__init__()
        self.embedding = nn.Embedding(num_entities, dim)

    def forward(self):
        return self.embedding.weight