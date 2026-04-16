import torch
import torch.nn as nn


class TKGScorer(nn.Module):
    def __init__(self, num_entities, num_relations, hidden_dim=128):
        super().__init__()

        self.entity_emb = nn.Embedding(num_entities, hidden_dim)
        self.relation_emb = nn.Embedding(num_relations, hidden_dim)

    def forward(self, h_s, r):
        """
        h_s: [batch_size, hidden_dim]
        r: [batch_size]
        """

        r_e = self.relation_emb(r)

        scores = torch.matmul(h_s + r_e, self.entity_emb.weight.T)

        return scores