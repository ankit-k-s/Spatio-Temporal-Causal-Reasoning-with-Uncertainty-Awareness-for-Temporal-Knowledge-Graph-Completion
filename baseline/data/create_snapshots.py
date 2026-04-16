import torch
from collections import defaultdict


def load_data(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            s, r, o, t, _ = map(int, line.strip().split())
            data.append((s, r, o, t))
    return data


def create_snapshots(data):
    snapshots = defaultdict(list)

    # group by time
    for s, r, o, t in data:
        snapshots[t].append((s, r, o))

    graph_snapshots = {}

    for t in snapshots:
        edges = snapshots[t]

        src = []
        dst = []
        rel = []

        for s, r, o in edges:
            src.append(s)
            dst.append(o)
            rel.append(r)

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_type = torch.tensor(rel, dtype=torch.long)

        graph_snapshots[t] = (edge_index, edge_type)

    return graph_snapshots


if __name__ == "__main__":
    file_path = "../../data/icews14/train_split.txt"

    data = load_data(file_path)
    snapshots = create_snapshots(data)

    print("Total snapshots:", len(snapshots))

    # print one example
    sample_t = list(snapshots.keys())[0]
    edge_index, edge_type = snapshots[sample_t]

    print(f"\nSample time: {sample_t}")
    print("edge_index shape:", edge_index.shape)
    print("edge_type shape:", edge_type.shape)

    # ========================
    # SAVE SNAPSHOTS
    # ========================
    save_path = "../snapshots.pt"

    torch.save(snapshots, save_path)

    print(f"\nSnapshots saved at: {save_path}")