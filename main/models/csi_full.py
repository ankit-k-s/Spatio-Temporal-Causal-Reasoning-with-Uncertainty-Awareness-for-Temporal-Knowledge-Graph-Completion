import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# EDGE MASK GENERATOR
# =========================
class EdgeMaskGenerator(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(4 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

    def forward(self, h, edge_index, edge_type, edge_time,
                rel_emb, time_emb, query_rel):

        src, _ = edge_index

        h_s = h[src]
        h_r = rel_emb(edge_type)
        h_t = time_emb(edge_time)

        h_rq = rel_emb(query_rel).unsqueeze(0).repeat(len(edge_type), 1)

        x = torch.cat([h_s, h_r, h_rq, h_t], dim=1)

        alpha = self.mlp(x)
        mask = torch.sigmoid(alpha)

        return mask.squeeze()


# =========================
# CSI MODEL
# =========================
class CSIFull(nn.Module):
    def __init__(self, encoder_c, encoder_s,
                 base_encoder_c, base_encoder_s,
                 num_entities, dim):
        super().__init__()

        # Temporal encoders
        self.encoder_c = encoder_c
        self.encoder_s = encoder_s

        # Base encoders (embeddings only)
        self.base_c = base_encoder_c
        self.base_s = base_encoder_s

        self.mask_gen = EdgeMaskGenerator(dim)

        self.pred_c = nn.Linear(dim, num_entities)
        self.pred_s = nn.Linear(dim, num_entities)
        self.pred_do = nn.Linear(dim, num_entities)

    # =========================
    # TRAINING FORWARD (UNCHANGED)
    # =========================
    def forward(self, edge_index, edge_type, edge_time, query_rel):

        h_init = self.base_c.entity_emb.weight

        # MASK
        M = self.mask_gen(
            h_init,
            edge_index,
            edge_type,
            edge_time,
            self.base_c.rel_emb,
            self.base_c.time_emb,
            query_rel
        )

        M_bar = 1 - M

        # ENCODERS (MASKED)
        hc = self.encoder_c(
            edge_index, edge_type, edge_time, edge_weight=M
        )

        hs = self.encoder_s(
            edge_index, edge_type, edge_time, edge_weight=M_bar
        )

        # PREDICTIONS
        pc = self.pred_c(hc)
        ps = self.pred_s(hs)

        # INTERVENTION
        perm = torch.randperm(hs.size(0), device=hs.device)
        hs_perm = hs[perm]

        h_do = hc + hs_perm
        p_do = self.pred_do(h_do)

        return pc, ps, p_do, hc, hs

    # =========================
    # FAST BASE ENCODING (EVAL)
    # =========================
    def encode_base(self, edge_index, edge_type, edge_time):
        """
        Relation-independent encoding
        """

        hc_base = self.encoder_c(
            edge_index,
            edge_type,
            edge_time,
            edge_weight=None
        )

        hs_base = self.encoder_s(
            edge_index,
            edge_type,
            edge_time,
            edge_weight=None
        )

        return hc_base, hs_base

    # =========================
    # FAST SCORING (EVAL)
    # =========================
    def score(self, hc_base, hs_base,
              edge_index, edge_type, edge_time, query_rel):
        """
        Fast CSI scoring without recomputing encoder
        """

        h_init = self.base_c.entity_emb.weight

        # MASK (still relation dependent)
        M = self.mask_gen(
            h_init,
            edge_index,
            edge_type,
            edge_time,
            self.base_c.rel_emb,
            self.base_c.time_emb,
            query_rel
        )

        # NOTE:
        # We DO NOT re-run encoder with mask (expensive)
        hc = hc_base
        hs = hs_base

        # PREDICTIONS
        pc = self.pred_c(hc)
        ps = self.pred_s(hs)

        # INTERVENTION
        perm = torch.randperm(hs.size(0), device=hs.device)
        hs_perm = hs[perm]

        h_do = hc + hs_perm
        p_do = self.pred_do(h_do)

        return pc, ps, p_do


# =========================
# RANKING LOSS
# =========================
def ranking_loss(pos_scores, neg_scores):
    pos = pos_scores.unsqueeze(1)
    diff = pos - neg_scores
    return torch.nn.functional.softplus(-diff).mean()


# =========================
# CSI LOSSES
# =========================
def loss_sup(pc, target):
    return F.cross_entropy(pc, target)


def loss_uniform(ps):
    uniform = torch.full_like(ps, 1.0 / ps.size(1))
    return F.kl_div(F.log_softmax(ps, dim=1), uniform, reduction='batchmean')


def loss_causal(p_do, target):
    return F.cross_entropy(p_do, target)


def total_loss(pc, ps, p_do, target, lambda1=0.1, lambda2=0.1):
    return (
        loss_sup(pc, target)
        + lambda1 * loss_uniform(ps)
        + lambda2 * loss_causal(p_do, target)
    )