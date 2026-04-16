import torch
import torch.nn as nn
from models.ssm import DiagonalSISOCell


class GraphSSM(nn.Module):
    def __init__(self, dim, d_state=16):
        super().__init__()

        self.cell = DiagonalSISOCell(d_state=d_state, d_input=dim)

    def forward(self, node_embeddings_seq):
        state = None
        outputs = []

        for x_t in node_embeddings_seq:
            out = self.cell(x_t, state)  # (N, D, d_state)

            # Reduce state dimension → (N, D)
            out = out.mean(dim=-1)

            # Update state
            state = out.unsqueeze(-1).repeat(1, 1, self.cell.d_state)

            outputs.append(out)

        return torch.stack(outputs, dim=0)