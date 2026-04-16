import torch
import sys
import os

sys.path.append(os.path.abspath("../external/graph_ssm"))

class GraphSSMWrapper:
    def __init__(self, config):
        pass

    def forward(self, sequence):
        """
        sequence: list of node embeddings [T, N, D]
        """

        # Convert to their input format

        output = sequence  # replace later

        return output