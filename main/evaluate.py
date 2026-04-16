import torch

from models.graph_mamba import GraphMamba
from models.csi_full import CSIFull
from data.toy_dataset import ToyTemporalKG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, dataset, k_list=[1, 3, 10]):
    model.eval()

    edge_index = dataset.edge_index.to(device)
    edge_type = dataset.edge_type.to(device)
    edge_time = dataset.edge_time.to(device)

    ranks = []

    with torch.no_grad():
        for i in range(edge_index.size(1)):

            h = edge_index[0, i]
            t = edge_index[1, i]
            r = edge_type[i]

            #  Query-specific forward (VERY IMPORTANT)
            pc, _, _, _, _ = model(
                edge_index,
                edge_type,
                edge_time,
                query_rel=r
            )

            scores = pc[h]

            _, indices = torch.sort(scores, descending=True)

            rank = (indices == t).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)

    ranks = torch.tensor(ranks).float()

    # =========================
    # METRICS
    # =========================
    mrr = torch.mean(1.0 / ranks)

    hits = {}
    for k in k_list:
        hits[k] = torch.mean((ranks <= k).float())

    print(f"\nEvaluation Results:")
    print(f"MRR: {mrr:.4f}")
    for k in k_list:
        print(f"Hits@{k}: {hits[k]:.4f}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    dataset = ToyTemporalKG()

    dim = 64

    encoder_c = GraphMamba(
        dataset.num_entities,
        dataset.num_relations,
        dataset.num_timestamps,
        dim
    ).to(device)

    encoder_s = GraphMamba(
        dataset.num_entities,
        dataset.num_relations,
        dataset.num_timestamps,
        dim
    ).to(device)

    model = CSIFull(encoder_c, encoder_s, dataset.num_entities, dim).to(device)

    #  LOAD TRAINED MODEL
    model.load_state_dict(torch.load("csi_model.pt", map_location=device))

    evaluate(model, dataset)