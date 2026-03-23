import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.node_embeddings import NodeEmbedding
from models.graph_mamba import GraphMamba
from models.graph_ssm import GraphSSM
from models.predictor import TKGScorer


# ========================
# LOAD DATA
# ========================
snapshots = torch.load("../snapshots.pt")
times = sorted(snapshots.keys())

num_entities = 12482
num_relations = 258
hidden_dim = 128


# ========================
# INIT MODELS
# ========================
node_emb = NodeEmbedding(num_entities, hidden_dim)
graph_mamba = GraphMamba(hidden_dim)
graph_ssm = GraphSSM(hidden_dim)
predictor = TKGScorer(num_entities, num_relations, hidden_dim)


# ========================
# BUILD TEMPORAL STATES
# ========================
X = node_emb()
H_sequence = []

for t in times[:10]:
    edge_index, edge_type = snapshots[t]
    H_t = graph_mamba(X, edge_index)
    H_sequence.append(H_t)

S_sequence = graph_ssm(H_sequence)


# ========================
# TAKE LAST TIME STEP
# ========================
S_t = S_sequence[-1]


# ========================
# SAMPLE QUERY
# ========================
# example: (s, r, ?, t)
s = torch.tensor([10])
r = torch.tensor([5])

h_s = S_t[s]


# ========================
# PREDICTION
# ========================
scores = predictor(h_s, r)

print("Scores shape:", scores.shape)

topk = torch.topk(scores, k=5)

print("\nTop 5 predicted entity IDs:", topk.indices)