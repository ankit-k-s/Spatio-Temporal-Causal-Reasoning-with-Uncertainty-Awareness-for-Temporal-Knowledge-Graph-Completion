import torch
import sys
import os

sys.path.append(os.path.abspath("../external/graph_mamba"))

# You will import their actual model here
# (depends on repo structure)

class GraphMambaWrapper:
    def __init__(self, config):
        # TODO: load actual GraphMamba model
        pass

    def forward(self, node_features, edge_index):
        """
        node_features: (N, D)
        edge_index: (2, E)
        """

        # Convert to their expected format
        # (this depends on their repo)

        # Example placeholder:
        output = node_features  # replace with real call

        return output