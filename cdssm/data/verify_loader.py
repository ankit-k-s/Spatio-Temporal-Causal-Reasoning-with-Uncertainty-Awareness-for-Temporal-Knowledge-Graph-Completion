import torch
from loader import TKGDataloader


def verify_loader(data_dir="ICEWS14"):
    loader = TKGDataloader(data_dir=data_dir)

    print("\n" + "=" * 60)
    print("RUNNING DETAILED CP-1 VERIFICATION")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Basic stats
    # --------------------------------------------------
    print("\n[1] BASIC STATS")
    print(f"Entities              : {loader.num_entities}")
    print(f"Relations (original)  : {loader.num_relations_original}")
    print(f"Relations (total)     : {loader.num_relations_total}")

    # --------------------------------------------------
    # 2. Timestamp ordering
    # --------------------------------------------------
    print("\n[2] TIMESTAMP ORDER CHECK")

    train_times = sorted(loader.train_snapshots.keys())
    valid_times = sorted(loader.valid_snapshots.keys())
    test_times = sorted(loader.test_snapshots.keys())

    print(f"Train range: {train_times[0]} → {train_times[-1]}")
    print(f"Valid range: {valid_times[0]} → {valid_times[-1]}")
    print(f"Test  range: {test_times[0]} → {test_times[-1]}")

    assert train_times[-1] <= valid_times[0], "Train overlaps with Valid"
    assert valid_times[-1] <= test_times[0], "Valid overlaps with Test"

    print("Temporal split is correct")

    # --------------------------------------------------
    # 3. Time ID normalization
    # --------------------------------------------------
    print("\n[3] TIME ID NORMALIZATION CHECK")

    all_time_ids = []
    for split in [
        loader.train_snapshots,
        loader.valid_snapshots,
        loader.test_snapshots,
    ]:
        for tau in split:
            _, _, time_id = split[tau]
            all_time_ids.append(time_id)

    min_id = min(all_time_ids)
    max_id = max(all_time_ids)

    print(f"Time ID range: {min_id} → {max_id}")

    assert min_id == 0, "Time IDs do not start from 0"
    assert len(set(all_time_ids)) == (max_id + 1), "Time IDs are not continuous"

    print("Time IDs normalized correctly")

    # --------------------------------------------------
    # 4. Edge structure check
    # --------------------------------------------------
    print("\n[4] EDGE STRUCTURE CHECK")

    sample_tau = train_times[0]
    edge_index, edge_type, time_id = loader.train_snapshots[sample_tau]

    print(f"Sample tau: {sample_tau}")
    print(f"edge_index shape: {edge_index.shape}")
    print(f"edge_type shape : {edge_type.shape}")

    assert edge_index.shape[0] == 2, "edge_index must have shape [2, E]"
    assert edge_index.shape[1] == edge_type.shape[0], "Edge count mismatch"

    print("Edge structure valid")

    # --------------------------------------------------
    # 5. Entity bounds check
    # --------------------------------------------------
    print("\n[5] ENTITY INDEX CHECK")

    max_node = edge_index.max().item()
    print(f"Max node index: {max_node}")
    print(f"Total entities: {loader.num_entities}")

    assert max_node < loader.num_entities, "Node index exceeds entity count"

    print("Entity indices valid")

    # --------------------------------------------------
    # 6. Relation bounds check
    # --------------------------------------------------
    print("\n[6] RELATION INDEX CHECK")

    max_rel = edge_type.max().item()
    print(f"Max relation index: {max_rel}")

    assert max_rel < loader.num_relations_total, "Relation index out of range"

    print("Relation indices valid")

    # --------------------------------------------------
    # 7. Inverse edge consistency check
    # --------------------------------------------------
    print("\n[7] INVERSE EDGE CHECK (PAIRWISE)")

    edge_index, edge_type, _ = loader.train_snapshots[train_times[0]]

    num_edges = edge_index.shape[1]

    correct = 0
    total = num_edges // 2

    for i in range(0, num_edges, 2):
        h1, t1 = edge_index[0, i], edge_index[1, i]
        r1 = edge_type[i]

        h2, t2 = edge_index[0, i+1], edge_index[1, i+1]
        r2 = edge_type[i+1]

        if h1 == t2 and t1 == h2:
            correct += 1

    print(f"Correct inverse pairs: {correct}/{total}")

    assert correct > total * 0.95, "Inverse edges incorrectly constructed"

    print("Inverse edges verified correctly")

    # --------------------------------------------------
    # 8. Graph density check
    # --------------------------------------------------
    print("\n[8] GRAPH DENSITY CHECK")

    avg_edges = sum(
        [snap[0].shape[1] for snap in loader.train_snapshots.values()]
    ) / len(loader.train_snapshots)

    print(f"Average edges per snapshot: {int(avg_edges)}")

    assert avg_edges > 100, "Graph appears too sparse"

    print("Graph density is reasonable")

    # --------------------------------------------------
    # Final
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    verify_loader()