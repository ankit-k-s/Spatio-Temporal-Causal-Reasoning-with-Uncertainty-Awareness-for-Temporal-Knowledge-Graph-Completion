import torch
import sys

sys.path.append(".")

from models.graph_mamba import GraphMamba
from models.csi_model import CSIFullModel

# config
num_entities = 100
num_relations = 10
num_timestamps = 20
dim = 64
num_edges = 300

# fake data
edge_index = torch.randint(0, num_entities, (2, num_edges))
edge_type = torch.randint(0, num_relations, (num_edges,))
edge_time = torch.randint(0, num_timestamps, (num_edges,))

targets = torch.randint(0, num_entities, (num_entities,))

# encoder
encoder = GraphMamba(
    num_entities,
    num_relations,
    num_timestamps,
    dim
)

# CSI model
model = CSIFullModel(encoder, dim, num_entities)

# forward
scores, gamma, hc, hs = model(edge_index, edge_type, edge_time)

print("Scores:", scores.shape)
print("Gamma:", gamma.shape)