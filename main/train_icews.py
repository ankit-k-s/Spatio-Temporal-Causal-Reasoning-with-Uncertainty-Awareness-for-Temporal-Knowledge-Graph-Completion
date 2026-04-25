import torch
import torch.optim as optim

from models.graph_mamba import GraphMamba
from models.temporal_encoder import TemporalEncoder
from models.csi_full import CSIFull, ranking_loss
from utils.negative_sampling import sample_negatives
from data.icews_loader import ICEWS14Dataset

torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# DATA
# =========================
dataset = ICEWS14Dataset("data")

edge_index = dataset.edge_index.to(device)
edge_type = dataset.edge_type.to(device)
edge_time = dataset.edge_time.to(device)

# =========================
# MODEL
# =========================
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

# TEMPORAL ENCODERS (GraphSSM inside)
encoder_c = TemporalEncoder(base_c, dim).to(device)
encoder_s = TemporalEncoder(base_s, dim).to(device)

# FINAL MODEL
model = CSIFull(
    encoder_c,
    encoder_s,
    base_c,
    base_s,
    dataset.num_entities,
    dim
).to(device)

# =========================
# OPTIMIZER
# =========================
optimizer = optim.Adam(model.parameters(), lr=5e-4)

# =========================
# SCHEDULER
# =========================
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5
)

# =========================
# MIXED PRECISION (H100)
# =========================
scaler = torch.amp.GradScaler("cuda")

# =========================
# TRAIN SETTINGS
# =========================
batch_size = 1024
num_neg = 64
epochs = 100

best_loss = float("inf")

# =========================
# EARLY STOPPING
# =========================
patience = 10
no_improve = 0

# =========================
# TRAIN LOOP
# =========================
for epoch in range(epochs):
    model.train()

    heads, rels, tails, times = dataset.sample_batch(batch_size)

    heads = heads.to(device)
    rels = rels.to(device)
    tails = tails.to(device)

    query_rel = rels[0]

    # =========================
    # FORWARD + LOSS (AMP)
    # =========================
    with torch.amp.autocast("cuda"):

        pc, ps, p_do, hc, hs = model(
            edge_index,
            edge_type,
            edge_time,
            query_rel=query_rel
        )

        # NEGATIVE SAMPLING
        neg_tails = sample_negatives(
            dataset.num_entities,
            batch_size,
            num_neg
        ).to(device)

        # RANKING LOSS
        pos_scores = pc[heads, tails]
        neg_scores = pc[heads.unsqueeze(1), neg_tails]

        loss_rank = ranking_loss(pos_scores, neg_scores)

        # CSI LOSSES
        ps_batch = ps[heads]
        uniform = torch.full_like(ps_batch, 1.0 / ps_batch.size(1))

        loss_uniform = torch.nn.functional.kl_div(
            torch.log_softmax(ps_batch, dim=1),
            uniform,
            reduction='batchmean'
        )

        pdo_batch = p_do[heads]
        loss_causal = torch.nn.functional.cross_entropy(pdo_batch, tails)

        # FINAL LOSS
        loss = loss_rank + 0.01 * loss_uniform + 0.01 * loss_causal

    # =========================
    # BACKPROP
    # =========================
    optimizer.zero_grad()

    scaler.scale(loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    # =========================
    # SCHEDULER STEP
    # =========================
    scheduler.step(loss.item())

    print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    # =========================
    # SAVE BEST MODEL + EARLY STOP
    # =========================
    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(model.state_dict(), "best_icews_model.pt")
        print(" Best model saved!")
        no_improve = 0
    else:
        no_improve += 1

    if no_improve >= patience:
        print(" Early stopping triggered")
        break