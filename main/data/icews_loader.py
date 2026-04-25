import torch


class ICEWS14Dataset:
    def __init__(self, path):
        self.path = path

        # =========================
        # LOAD FILES
        # =========================
        self.train_data = self.load_file("train_split.txt")
        self.valid_data = self.load_file("valid.txt")
        self.test_data = self.load_file("test.txt")

        # =========================
        # BUILD MAPPINGS
        # =========================
        self.build_mappings()

        # =========================
        # BUILD GRAPH
        # =========================
        self.edge_index, self.edge_type, self.edge_time = self.get_graph()

        # =========================
        # BUILD FILTER DICT ( IMPORTANT)
        # =========================
        self.build_filter_dict()

    # =========================
    # LOAD RAW FILE
    # =========================
    def load_file(self, filename):
        data = []

        with open(f"{self.path}/{filename}") as f:
            for line in f:
                parts = line.strip().split()

                # robust parsing
                h = parts[0]
                r = parts[1]
                t = parts[2]
                ts = parts[3]

                data.append((h, r, t, ts))

        return data

    # =========================
    # BUILD ID MAPPINGS
    # =========================
    def build_mappings(self):
        entities = set()
        relations = set()
        timestamps = set()

        for dataset in [self.train_data, self.valid_data, self.test_data]:
            for h, r, t, ts in dataset:
                entities.add(h)
                entities.add(t)
                relations.add(r)
                timestamps.add(ts)

        self.entity2id = {e: i for i, e in enumerate(sorted(entities))}
        self.rel2id = {r: i for i, r in enumerate(sorted(relations))}
        self.time2id = {t: i for i, t in enumerate(sorted(timestamps))}

        self.num_entities = len(self.entity2id)
        self.num_relations = len(self.rel2id)
        self.num_timestamps = len(self.time2id)

        # encode all splits
        self.train_triples = self.encode(self.train_data)
        self.valid_triples = self.encode(self.valid_data)
        self.test_triples = self.encode(self.test_data)

    # =========================
    # ENCODE TRIPLES
    # =========================
    def encode(self, data):
        triples = []

        for h, r, t, ts in data:
            triples.append((
                self.entity2id[h],
                self.rel2id[r],
                self.entity2id[t],
                self.time2id[ts]
            ))

        return torch.tensor(triples, dtype=torch.long)

    # =========================
    # BUILD GRAPH (TRAIN ONLY)
    # =========================
    def get_graph(self):
        triples = self.train_triples

        edge_index = triples[:, [0, 2]].t()  # (2, E)
        edge_type = triples[:, 1]            # (E,)
        edge_time = triples[:, 3]            # (E,)

        return edge_index, edge_type, edge_time

    # =========================
    # FILTER DICTIONARY ( KEY PART)
    # =========================
    def build_filter_dict(self):
        self.filter_dict = {}

        # combine all splits
        all_triples = torch.cat([
            self.train_triples,
            self.valid_triples,
            self.test_triples
        ], dim=0)

        for h, r, t, ts in all_triples:
            key = (h.item(), r.item(), ts.item())

            if key not in self.filter_dict:
                self.filter_dict[key] = set()

            self.filter_dict[key].add(t.item())

    # =========================
    # SAMPLING (TRAIN)
    # =========================
    def sample_batch(self, batch_size):
        idx = torch.randint(0, len(self.train_triples), (batch_size,))
        batch = self.train_triples[idx]

        heads = batch[:, 0]
        rels = batch[:, 1]
        tails = batch[:, 2]
        times = batch[:, 3]

        return heads, rels, tails, times