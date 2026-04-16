import torch
from models.graph_mamba import GraphMamba
from models.csi_full import CSIFull

num_entities = 100
num_relations = 10
num_timestamps = 20
dim = 64
num_edges = 300

edge_index = torch.randint(0, num_entities, (2, num_edges))
edge_type = torch.randint(0, num_relations, (num_edges,))
edge_time = torch.randint(0, num_timestamps, (num_edges,))
target = torch.randint(0, num_entities, (num_entities,))

query_rel = torch.tensor(2)

encoder_c = GraphMamba(num_entities, num_relations, num_timestamps, dim)
encoder_s = GraphMamba(num_entities, num_relations, num_timestamps, dim)

model = CSIFull(encoder_c, encoder_s, num_entities, dim)

pc, ps, p_do, hc, hs = model(
    edge_index,
    edge_type,
    edge_time,
    query_rel
)

print("pc:", pc.shape)
print("ps:", ps.shape)
print("p_do:", p_do.shape)