import torch

class ToyTemporalKG:
    def __init__(self, num_entities=100, num_relations=10, num_timestamps=20, num_edges=1000):
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.num_timestamps = num_timestamps

        self.edge_index = torch.randint(0, num_entities, (2, num_edges))
        self.edge_type = torch.randint(0, num_relations, (num_edges,))
        self.edge_time = torch.randint(0, num_timestamps, (num_edges,))

    def sample_batch(self, batch_size=64):
        idx = torch.randint(0, self.edge_index.size(1), (batch_size,))

        heads = self.edge_index[0, idx]
        rels = self.edge_type[idx]
        tails = self.edge_index[1, idx]
        times = self.edge_time[idx]

        return heads, rels, tails, times