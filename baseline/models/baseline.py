import torch
import torch.nn as nn

class BaselineModel(nn.Module):
    def __init__(self, num_entities, num_relations, hidden_dim=128):
        super().__init__()
        
        self.entity_emb = nn.Embedding(num_entities, hidden_dim)
        self.relation_emb = nn.Embedding(num_relations, hidden_dim)

    def forward(self, s, r):
        h_s = self.entity_emb(s)         # [1, hidden_dim]
        r_e = self.relation_emb(r)       # [1, hidden_dim]
        
        scores = torch.matmul(h_s + r_e, self.entity_emb.weight.T)
        return scores