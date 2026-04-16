import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.node_embeddings import NodeEmbedding
from models.graph_mamba import GraphMamba


# ========================
# LOAD SNAPSHOTS
# ========================
snapshots = torch.load("../snapshots.pt")

t = list(snapshots.keys())[0]
edge_index, edge_type = snapshots[t]


# ========================
# INIT EMBEDDINGS
# ========================
num_entities = 12482
hidden_dim = 128

node_emb = NodeEmbedding(num_entities, hidden_dim)
X = node_emb()


# ========================
# GRAPH MAMBA
# ========================
model = GraphMamba(hidden_dim)

H = model(X, edge_index)

print("Input shape:", X.shape)
print("Output shape:", H.shape)