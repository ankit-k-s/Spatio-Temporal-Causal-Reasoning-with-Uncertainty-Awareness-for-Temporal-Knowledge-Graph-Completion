import torch
import torch.optim as optim

from models.graph_mamba import GraphMamba
from models.csi_full import CSIFull, ranking_loss
from data.toy_dataset import ToyTemporalKG
from utils.negative_sampling import sample_negatives
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# DATA
# =========================
dataset = ToyTemporalKG()

# =========================
# MODEL
# =========================
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

optimizer = optim.Adam(model.parameters(), lr=5e-4)

# =========================
# TRAIN LOOP
# =========================
batch_size = 512        # was 64
num_neg = 64            # was 10
epochs = 50             # was 10

edge_index = dataset.edge_index.to(device)
edge_type = dataset.edge_type.to(device)
edge_time = dataset.edge_time.to(device)
best_loss = float("inf")
for epoch in range(epochs):
    model.train()

    heads, rels, tails, times = dataset.sample_batch(batch_size)

    heads = heads.to(device)
    rels = rels.to(device)
    tails = tails.to(device)

    query_rel = rels[0]

    # Forward
    pc, ps, p_do, hc, hs = model(
        edge_index,
        edge_type,
        edge_time,
        query_rel=query_rel
    )

    # =========================
    # NEGATIVE SAMPLING
    # =========================
    neg_tails = sample_negatives(dataset.num_entities, batch_size, num_neg).to(device)

    # =========================
    # RANKING LOSS (MAIN)
    # =========================
    pos_scores = pc[heads, tails]
    neg_scores = pc[heads.unsqueeze(1), neg_tails]

    loss_rank = ranking_loss(pos_scores, neg_scores)

    # =========================
    # CSI LOSSES
    # =========================

    # 1. Shortcut uniformity
    ps_batch = ps[heads]
    uniform = torch.full_like(ps_batch, 1.0 / ps_batch.size(1))

    loss_uniform = torch.nn.functional.kl_div(
        torch.log_softmax(ps_batch, dim=1),
        uniform,
        reduction='batchmean'
    )

    # 2. Intervention consistency
    pdo_batch = p_do[heads]
    loss_causal = torch.nn.functional.cross_entropy(pdo_batch, tails)

    # =========================
    # FINAL LOSS (CRITICAL WEIGHTS)
    # =========================
    lambda1 = 0.01   # VERY SMALL
    lambda2 = 0.01   # VERY SMALL

    loss = loss_rank + lambda1 * loss_uniform + lambda2 * loss_causal

    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # SAVE MODEL
    print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(model.state_dict(), "best_model.pt")
        print("Best model saved!")

    