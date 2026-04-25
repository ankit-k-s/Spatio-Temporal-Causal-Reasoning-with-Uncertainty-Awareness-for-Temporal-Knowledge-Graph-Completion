import torch
import time
from tqdm import tqdm

from models.graph_mamba import GraphMamba
from models.temporal_encoder import TemporalEncoder
from models.csi_full import CSIFull
from data.icews_loader import ICEWS14Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, dataset, num_samples=2000):
    model.eval()

    edge_index = dataset.edge_index.to(device)
    edge_type = dataset.edge_type.to(device)
    edge_time = dataset.edge_time.to(device)

    # =========================
    # SAMPLE TEST TRIPLES
    # =========================
    test_triples = dataset.test_triples[:num_samples]

    print(f"Using {len(test_triples)} test triples")

    ranks = []

    start_time = time.time()

    with torch.no_grad():

        # =========================
        # GROUP BY RELATION (IMPORTANT)
        # =========================
        rel_groups = {}
        for h, r, t, ts in test_triples:
            if r not in rel_groups:
                rel_groups[r] = []
            rel_groups[r].append((h, t, ts))

        print(f"Total relations: {len(rel_groups)}")

        # =========================
        # LOOP PER RELATION
        # =========================
        for r in tqdm(rel_groups.keys(), desc="Evaluating (tail only)"):

            query_rel = torch.tensor(r, device=device)

            # FULL MODEL FORWARD (IMPORTANT)
            pc, _, _, _, _ = model(
                edge_index,
                edge_type,
                edge_time,
                query_rel=query_rel
            )

            triples = rel_groups[r]

            for h, t, ts in triples:

                scores = pc[h].clone()

                # =========================
                # FILTERING (CRITICAL)
                # =========================
                key = (h, r, ts)
                if key in dataset.filter_dict:
                    for t_filt in dataset.filter_dict[key]:
                        if t_filt != t:
                            scores[t_filt] = -1e9

                # =========================
                # RANK
                # =========================
                _, indices = torch.sort(scores, descending=True)
                rank = (indices == t).nonzero(as_tuple=True)[0].item() + 1

                ranks.append(rank)

    # =========================
    # METRICS
    # =========================
    ranks = torch.tensor(ranks).float()

    mrr = torch.mean(1.0 / ranks).item()
    hits1 = torch.mean((ranks <= 1).float()).item()
    hits3 = torch.mean((ranks <= 3).float()).item()
    hits10 = torch.mean((ranks <= 10).float()).item()

    total_time = time.time() - start_time

    print("\n===== FINAL RESULTS (TAIL PREDICTION) =====")
    print(f"MRR: {mrr:.4f}")
    print(f"Hits@1: {hits1:.4f}")
    print(f"Hits@3: {hits3:.4f}")
    print(f"Hits@10: {hits10:.4f}")
    print(f"Time taken: {total_time/60:.2f} minutes")

    return {
        "MRR": mrr,
        "Hits@1": hits1,
        "Hits@3": hits3,
        "Hits@10": hits10
    }


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    dataset = ICEWS14Dataset("data")

    dim = 128

    # BASE ENCODERS
    base_c = GraphMamba(
        dataset.num_entities,
        dataset.num_relations,
        dataset.num_timestamps,
        dim
    ).to(device)

    base_s = GraphMamba(
        dataset.num_entities,
        dataset.num_relations,
        dataset.num_timestamps,
        dim
    ).to(device)

    # TEMPORAL ENCODERS
    encoder_c = TemporalEncoder(base_c, dim).to(device)
    encoder_s = TemporalEncoder(base_s, dim).to(device)

    # MODEL
    model = CSIFull(
        encoder_c,
        encoder_s,
        base_c,
        base_s,
        dataset.num_entities,
        dim
    ).to(device)

    # LOAD WEIGHTS
    model.load_state_dict(torch.load("best_icews_model.pt", map_location=device))

    evaluate(model, dataset, num_samples=2000)