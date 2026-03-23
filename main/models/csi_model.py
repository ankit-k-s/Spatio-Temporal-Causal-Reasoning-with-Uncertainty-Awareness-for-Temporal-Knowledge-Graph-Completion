import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# MASK GENERATOR
# =========================
class MaskGenerator(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

    def forward(self, h):
        gamma = torch.sigmoid(self.mlp(h))  # (N,1)
        return gamma


# =========================
# GRAPH SPLIT
# =========================
def split_graph(edge_index, edge_type, edge_time, gamma):
    src, dst = edge_index

    edge_score = (gamma[src] + gamma[dst]) / 2.0
    mask = edge_score.squeeze() > 0.5

    Gc = edge_index[:, mask]
    Gs = edge_index[:, ~mask]

    type_c = edge_type[mask]
    type_s = edge_type[~mask]

    time_c = edge_time[mask]
    time_s = edge_time[~mask]

    return Gc, type_c, time_c, Gs, type_s, time_s


# =========================
# INTERVENTION
# =========================
def intervention(hs):
    perm = torch.randperm(hs.size(0))
    return hs[perm]


# =========================
# PREDICTOR
# =========================
class Predictor(nn.Module):
    def __init__(self, dim, num_entities):
        super().__init__()
        self.linear = nn.Linear(dim, num_entities)

    def forward(self, h):
        return self.linear(h)


# =========================
# FULL CSI MODEL
# =========================
class CSIFullModel(nn.Module):
    def __init__(self, encoder, dim, num_entities):
        super().__init__()

        self.encoder = encoder
        self.mask_gen = MaskGenerator(dim)
        self.predictor = Predictor(dim, num_entities)

    def forward(self, edge_index, edge_type, edge_time):
        # -----------------------
        # 1. Encode full graph
        # -----------------------
        h = self.encoder(edge_index, edge_type, edge_time)

        # -----------------------
        # 2. Mask generation
        # -----------------------
        gamma = self.mask_gen(h)

        # -----------------------
        # 3. Split graph
        # -----------------------
        Gc, type_c, time_c, Gs, type_s, time_s = split_graph(
            edge_index, edge_type, edge_time, gamma
        )

        # -----------------------
        # 4. Dual encoding
        # -----------------------
        hc = self.encoder(Gc, type_c, time_c)
        hs = self.encoder(Gs, type_s, time_s)

        # -----------------------
        # 5. Intervention
        # -----------------------
        hs_do = intervention(hs)

        # -----------------------
        # 6. Combine
        # -----------------------
        h_final = hc + hs_do

        # -----------------------
        # 7. Prediction
        # -----------------------
        scores = self.predictor(h_final)

        return scores, gamma, hc, hs


# =========================
# LOSSES
# =========================
def loss_supervised(scores, targets):
    return F.cross_entropy(scores, targets)


def loss_uniform(hs):
    return torch.var(hs)


def loss_causal(hc, hs):
    return torch.mean((hc - hs) ** 2)


def total_loss(scores, targets, hc, hs, lambda1=0.1, lambda2=0.1):
    L_sup = loss_supervised(scores, targets)
    L_uni = loss_uniform(hs)
    L_cau = loss_causal(hc, hs)

    return L_sup + lambda1 * L_uni + lambda2 * L_cau