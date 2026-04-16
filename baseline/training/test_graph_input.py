import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.node_embeddings import NodeEmbedding


# ========================
# LOAD SNAPSHOTS
# ========================
snapshots = torch.load("../snapshots.pt")

print("Loaded snapshots:", len(snapshots))


# ========================
# INIT EMBEDDINGS
# ========================
num_entities = 12482  # from your earlier output
hidden_dim = 128

node_emb = NodeEmbedding(num_entities, hidden_dim)

X = node_emb()  # [num_entities, hidden_dim]

print("Node embedding shape:", X.shape)


# ========================
# TAKE ONE SNAPSHOT
# ========================
t = list(snapshots.keys())[0]

edge_index, edge_type = snapshots[t]

print("\nSample snapshot time:", t)
print("edge_index shape:", edge_index.shape)
print("edge_type shape:", edge_type.shape)


# ========================
# MAP EDGES → EMBEDDINGS
# ========================
src_nodes = edge_index[0]
dst_nodes = edge_index[1]

src_emb = X[src_nodes]
dst_emb = X[dst_nodes]

print("\nSource embedding shape:", src_emb.shape)
print("Destination embedding shape:", dst_emb.shape)