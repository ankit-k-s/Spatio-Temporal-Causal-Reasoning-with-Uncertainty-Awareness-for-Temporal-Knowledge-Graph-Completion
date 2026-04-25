import torch
import torch.nn as nn
from models.graph_ssm import GraphSSM


class TemporalEncoder(nn.Module):
    def __init__(self, base_encoder, dim):
        super().__init__()

        self.base_encoder = base_encoder
        self.ssm = GraphSSM(dim)

    def forward(self, edge_index, edge_type, edge_time, edge_weight=None):

        # =========================
        # SORT TIME
        # =========================
        unique_times = torch.unique(edge_time)
        unique_times = torch.sort(unique_times)[0]

        state = None
        last_out = None

        for t in unique_times:

            mask = (edge_time == t)

            ei_t = edge_index[:, mask]
            et_t = edge_type[mask]
            time_t = edge_time[mask]

            ew_t = edge_weight[mask] if edge_weight is not None else None

            # =========================
            # SPATIAL ENCODING
            # =========================
            h_t = self.base_encoder(ei_t, et_t, time_t, ew_t)

            # =========================
            # TEMPORAL UPDATE (SSM)
            # =========================
            out, state = self.ssm.cell(h_t, state)

            # VERY IMPORTANT (prevents memory explosion)
            state = state.detach()

            last_out = out

        return last_out  # (N, D)