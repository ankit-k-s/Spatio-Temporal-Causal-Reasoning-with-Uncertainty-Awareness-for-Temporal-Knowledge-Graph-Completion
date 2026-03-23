import torch
import sys
import os

sys.path.append(os.path.abspath("../external/diffutkg"))

class DiffusionWrapper:
    def __init__(self, config):
        pass

    def forward(self, h_query, relation):
        """
        h_query: (B, D)
        """

        # Replace predictor
        scores = torch.randn(h_query.size(0), 10000)  # placeholder

        return scores