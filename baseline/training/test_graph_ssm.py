import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.node_embeddings import NodeEmbedding
from models.graph_mamba import GraphMamba
from models.graph_ssm import GraphSSM


# ========================
# LOAD SNAPSHOTS
# ========================
snapshots = torch.load("../snapshots.pt")

# sort by time
times = sorted(snapshots.keys())


# ========================
# INIT
# ========================
num_entities = 12482
hidden_dim = 128

node_emb = NodeEmbedding(num_entities, hidden_dim)
graph_mamba = GraphMamba(hidden_dim)
graph_ssm = GraphSSM(hidden_dim)

X = node_emb()


# ========================
# BUILD H_sequence
# ========================
H_sequence = []

for t in times[:10]:  # use first 10 snapshots for testing
    edge_index, edge_type = snapshots[t]

    H_t = graph_mamba(X, edge_index)
    H_sequence.append(H_t)


# ========================
# TEMPORAL MODEL
# ========================
S_sequence = graph_ssm(H_sequence)

print("Number of time steps:", len(S_sequence))
print("Shape of one state:", S_sequence[0].shape)