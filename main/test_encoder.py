import torch
import sys

sys.path.append(".")

from models.graph_mamba import GraphMamba
from models.graph_ssm import GraphSSM

# -----------------------
# CONFIG
# -----------------------
num_entities = 100
num_relations = 10
num_timestamps = 20
dim = 64
num_edges = 300
T = 5

# -----------------------
# INIT MODELS
# -----------------------
graph_mamba = GraphMamba(
    num_entities,
    num_relations,
    num_timestamps,
    dim
)

graph_ssm = GraphSSM(dim)

# -----------------------
# TEST GRAPH MAMBA
# -----------------------
edge_index = torch.randint(0, num_entities, (2, num_edges))
edge_type = torch.randint(0, num_relations, (num_edges,))
edge_time = torch.randint(0, num_timestamps, (num_edges,))

x = graph_mamba(edge_index, edge_type, edge_time)

print("GraphMamba output:", x.shape)

# -----------------------
# TEMPORAL SEQUENCE
# -----------------------
sequence = []

for _ in range(T):
    edge_index = torch.randint(0, num_entities, (2, num_edges))
    edge_type = torch.randint(0, num_relations, (num_edges,))
    edge_time = torch.randint(0, num_timestamps, (num_edges,))

    x_t = graph_mamba(edge_index, edge_type, edge_time)
    sequence.append(x_t)

# -----------------------
# TEST GRAPH SSM
# -----------------------
out = graph_ssm(sequence)

print("GraphSSM output:", out.shape)