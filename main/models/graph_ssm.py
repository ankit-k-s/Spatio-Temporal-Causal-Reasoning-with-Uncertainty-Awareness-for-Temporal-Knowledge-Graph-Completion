import torch
import torch.nn as nn
from models.ssm import DiagonalSISOCell


class GraphSSM(nn.Module):
    def __init__(self, dim, d_state=8):
        super().__init__()

        self.cell = DiagonalSISOCell(d_state, dim)

    def forward(self, sequence):
        state = None
        outputs = []

        for x in sequence:
            out, state = self.cell(x, state)
            outputs.append(out)

        return torch.stack(outputs)