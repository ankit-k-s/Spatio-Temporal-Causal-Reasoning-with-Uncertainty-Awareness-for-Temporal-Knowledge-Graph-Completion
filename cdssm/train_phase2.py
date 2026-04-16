import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data.loader import TKGDataloader
from models.embedding import TKGEmbedding
from models.spatial import GraphMambaLayer
from models.temporal import GraphSSM
from models.causal import CausalDecomposer
from models.predictor import TKGCFusingPredictor
from models.diffusion import AsymmetricNoiseScheduler, BiSSMDenoiser


def train_phase_2():
    print("Initializing CD-SSM Phase 2 (D=200, TRUE FROZEN BACKBONE)...")

    loader = TKGDataloader(data_dir="data/ICEWS14")

    N = loader.num_entities
    NUM_REL = loader.num_relations_total
    NUM_TIME = len(loader.time2id)

    # -------------------------
    # STABLE CONFIG
    # -------------------------
    D = 200
    D_STATE = 16
    DENOISER_LAYERS = 1
    DENOISER_DROPOUT = 0.0

    LR = 5e-4
    EPOCHS = 100
    DIFFUSION_STEPS = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | D={D} | D_STATE={D_STATE}")

    # -------------------------
    # MODEL
    # -------------------------
    emb_layer = TKGEmbedding(N, NUM_REL, NUM_TIME, D).to(device)
    spatial_layer = GraphMambaLayer(d_model=D, seq_len=3).to(device)
    temporal_layer = GraphSSM(d_model=D, d_state=D_STATE, num_layers=2).to(device)
    decomposer = CausalDecomposer(d_model=D).to(device)
    predictor = TKGCFusingPredictor(d_model=D).to(device)

    scheduler = AsymmetricNoiseScheduler(
        num_timesteps=DIFFUSION_STEPS,
        elevated_beta=5.0
    ).to(device)

    denoiser = BiSSMDenoiser(
        d_model=D,
        d_state=D_STATE,
        num_timesteps=DIFFUSION_STEPS,
        num_layers=DENOISER_LAYERS,
        dropout=DENOISER_DROPOUT
    ).to(device)

    # 🔥 FIX: Phase 2 ONLY needs standard Cross Entropy to train the denoiser
    loss_fn = nn.CrossEntropyLoss().to(device)

    # -------------------------
    # LOAD PHASE 1
    # -------------------------
    try:
        checkpoint = torch.load("checkpoints/cdssm_phase1.pt", map_location=device, weights_only=True)

        emb_layer.load_state_dict(checkpoint['emb_layer'])
        spatial_layer.load_state_dict(checkpoint['spatial_layer'])
        temporal_layer.load_state_dict(checkpoint['temporal_layer'])
        decomposer.load_state_dict(checkpoint['decomposer'])
        predictor.load_state_dict(checkpoint['predictor'])

        print("✅ Phase 1 checkpoint loaded (D=200)")

    except FileNotFoundError:
        print("ERROR: Phase 1 checkpoint missing. Run Phase 1 with D=200 first.")
        return

    # -------------------------
    # 🔥 FIX: TRUE PHASE 1 FREEZING
    # -------------------------
    # 1. Put all Phase 1 modules in evaluation mode and turn off gradients
    for m in [emb_layer, spatial_layer, temporal_layer, decomposer, predictor]:
        m.eval()
        for param in m.parameters():
            param.requires_grad = False
            
    # 2. Put ONLY the denoiser in training mode
    denoiser.train()

    # 3. ONLY pass the denoiser parameters to the optimizer
    optimizer = AdamW(denoiser.parameters(), lr=LR, weight_decay=1e-4)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    os.makedirs("checkpoints", exist_ok=True)

    sorted_times = sorted(loader.train_snapshots.keys())

    # -------------------------
    # TRAIN LOOP
    # -------------------------
    for epoch in range(EPOCHS):

        print(f"\n--- Epoch {epoch+1}/{EPOCHS} | LR={optimizer.param_groups[0]['lr']:.2e} ---")

        states, prev_xs = None, None
        epoch_loss = 0.0

        pbar = tqdm(range(len(sorted_times) - 1), desc="Strict Forecasting")

        for i in pbar:

            tau_curr = sorted_times[i]
            tau_next = sorted_times[i + 1]

            edge_curr, type_curr, time_curr = loader.train_snapshots[tau_curr]
            edge_next, type_next, time_next = loader.train_snapshots[tau_next]

            edge_curr, type_curr = edge_curr.to(device), type_curr.to(device)
            edge_next, type_next = edge_next.to(device), type_next.to(device)

            time_curr_tensor = torch.tensor([time_curr], device=device)
            time_next_tensor = torch.tensor([time_next], device=device)

            optimizer.zero_grad()

            with torch.no_grad(): # Ensure absolutely no gradients flow through Phase 1 operations
                # -------------------------
                # HISTORY ENCODING
                # -------------------------
                x_base = emb_layer.get_all_entity_embeddings(time_curr_tensor).squeeze(0)
                x_spatial = spatial_layer(x_base, edge_curr, type_curr, emb_layer.rel_emb)
                H_curr, states, prev_xs = temporal_layer(x_spatial, states, prev_xs)

                # -------------------------
                # FORECAST TARGET
                # -------------------------
                subjects = edge_next[0]
                objects = edge_next[1]
                relations = type_next

                subj_states = H_curr[subjects]
                rel_embeddings = emb_layer.rel_emb(relations)

                t_emb = emb_layer.time_emb(time_next_tensor).squeeze(0)
                t_emb = t_emb.expand(len(subjects), -1)

                s_T = states[-1][subjects]
                s_T = F.normalize(s_T, dim=-1)

                # -------------------------
                # CAUSAL DECOMPOSITION
                # -------------------------
                h_c, h_s, mask = decomposer(subj_states, rel_embeddings)
                
                # -------------------------
                # DIFFUSION INPUTS
                # -------------------------
                m = torch.randint(0, DIFFUSION_STEPS, (len(subjects),), device=device)
                hc_noisy, _ = scheduler.forward_noise(h_c, m, "causal")

            # -------------------------
            # DENOISING (Gradients ON for denoiser only)
            # -------------------------
            hc_pred = denoiser(hc_noisy, rel_embeddings, s_T, t_emb, m)

            # -------------------------
            # SCORING & LOSS
            # -------------------------
            # We use the frozen predictor to evaluate the denoiser's output
            scores_c = predictor(hc_pred, rel_embeddings, H_curr)

            # Standard cross entropy just to train the denoiser to point to the right answer
            loss = loss_fn(scores_c, objects)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0) # Only clip denoiser
            optimizer.step()

            epoch_loss += loss.item()

            # detach temporal memory
            states = [s.detach() for s in states]
            prev_xs = [x.detach() for x in prev_xs]

            pbar.set_postfix({
                "Loss": f"{loss.item():.2f}",
                "Mask": f"{mask.mean().item():.2f}" # This will now stay safely static
            })

        lr_scheduler.step()

        print(f"Epoch {epoch+1} Avg Loss: {epoch_loss / (len(sorted_times)-1):.4f}")

    # -------------------------
    # SAVE
    # -------------------------
    print("\nSaving Phase 2 model...")

    checkpoint['denoiser'] = denoiser.state_dict()

    torch.save(checkpoint, "checkpoints/cdssm_phase2.pt")

    print("✅ Saved to checkpoints/cdssm_phase2.pt")


if __name__ == "__main__":
    train_phase_2()