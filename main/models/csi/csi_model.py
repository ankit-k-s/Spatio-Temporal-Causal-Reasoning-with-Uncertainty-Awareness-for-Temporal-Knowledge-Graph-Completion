import torch
import torch.nn as nn

from .attention import CSIAttention
from .splitter import split_graph
from .intervention import intervene

class CSIModel(nn.Module):
    def __init__(self, dim, mamba, ssm, predictor):
        super().__init__()
        self.attn = CSIAttention(dim)
        self.mamba = mamba
        self.ssm = ssm
        self.pred = predictor

    def forward(self, X, edge_index, edge_type, query, relation):

        src, dst = edge_index
        edge_embs = X[src] + X[dst]

        attn = self.attn(query, edge_embs).squeeze(0)

        (c_edge, _), (s_edge, _) = split_graph(edge_index, edge_type, attn)

        Hc = self.mamba(X, c_edge)
        Sc = self.ssm([Hc])[-1]

        Hs = self.mamba(X, s_edge)
        Ss = self.ssm([Hs])[-1]

        Ss_i = intervene(Ss)

        pred_c = self.pred(Sc, relation)
        pred_s = self.pred(Ss, relation)
        pred_i = self.pred(Sc + Ss_i, relation)

        return pred_c, pred_s, pred_i