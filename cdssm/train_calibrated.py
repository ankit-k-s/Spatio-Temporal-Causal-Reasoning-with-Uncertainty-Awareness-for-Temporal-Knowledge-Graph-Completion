import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from data.loader import TKGDataloader
from models.embedding import TKGEmbedding
from models.spatial import GraphMambaLayer
from models.temporal import GraphSSM
from models.causal import CausalDecomposer
from models.predictor import TKGCFusingPredictor
from models.diffusion import AsymmetricNoiseScheduler, BiSSMDenoiser
from models.calibration import ConfidenceHead

def compute_frequency_prior(loader, N, device):
    entity_freq = torch.zeros(N, device=device)
    for tau, (edge_index, _, _) in loader.train_snapshots.items():
        objects = edge_index[1].to(device)
        entity_freq.scatter_add_(0, objects, torch.ones_like(objects, dtype=torch.float))
    entity_freq = entity_freq / entity_freq.sum()
    return torch.log(entity_freq + 1e-9)

def train_calibrated():
    print("Initializing Calibrated Fine-Tuning (SAFE MODE)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = TKGDataloader(data_dir="data/ICEWS14")
    
    N, D, D_STATE, DIFFUSION_STEPS = loader.num_entities, 200, 16, 100
    LAMBDA_FREQ = 0.15 
    EPOCHS = 20 # Only need 20 epochs for fine-tuning
    
    freq_bias = compute_frequency_prior(loader, N, device)

    # 1. Load Architecture
    emb_layer = TKGEmbedding(N, loader.num_relations_total, len(loader.time2id), D).to(device)
    spatial_layer = GraphMambaLayer(d_model=D, seq_len=3).to(device)
    temporal_layer = GraphSSM(d_model=D, d_state=D_STATE, num_layers=2).to(device)
    decomposer = CausalDecomposer(d_model=D).to(device)
    predictor = TKGCFusingPredictor(d_model=D).to(device)
    scheduler = AsymmetricNoiseScheduler(num_timesteps=DIFFUSION_STEPS).to(device)
    denoiser = BiSSMDenoiser(d_model=D, d_state=D_STATE, num_timesteps=DIFFUSION_STEPS).to(device)
    
    # NEW: The Trainable Calibration Head
    confidence_head = ConfidenceHead(d_model=D).to(device)

    # 2. Load Existing Good Weights
    cp1 = torch.load("checkpoints/cdssm_phase1.pt", map_location=device, weights_only=True)
    emb_layer.load_state_dict(cp1['emb_layer'])
    spatial_layer.load_state_dict(cp1['spatial_layer'])
    temporal_layer.load_state_dict(cp1['temporal_layer'])
    decomposer.load_state_dict(cp1['decomposer'])
    predictor.load_state_dict(cp1['predictor'])
    
    cp2 = torch.load("checkpoints/cdssm_phase2.pt", map_location=device, weights_only=True)
    denoiser.load_state_dict(cp2['denoiser'])

    # 3. Freeze Backbone completely
    for m in [emb_layer, spatial_layer, temporal_layer, decomposer, predictor]:
        m.eval()
        for param in m.parameters(): param.requires_grad = False

    denoiser.train()
    confidence_head.train()

    # Train BOTH the denoiser and the new confidence head
    optimizer = AdamW(list(denoiser.parameters()) + list(confidence_head.parameters()), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss().to(device)

    sorted_times = sorted(loader.train_snapshots.keys())

    # 4. Fine-Tuning Loop
    for epoch in range(EPOCHS):
        states, prev_xs, epoch_loss = None, None, 0.0
        pbar = tqdm(range(len(sorted_times) - 1), desc=f"Fine-tuning Epoch {epoch+1}/{EPOCHS}")

        for i in pbar:
            tau_curr, tau_next = sorted_times[i], sorted_times[i + 1]
            edge_curr, type_curr, time_curr = loader.train_snapshots[tau_curr]
            edge_next, type_next, time_next = loader.train_snapshots[tau_next]

            edge_curr, type_curr = edge_curr.to(device), type_curr.to(device)
            edge_next, type_next = edge_next.to(device), type_next.to(device)
            time_curr_tensor = torch.tensor([time_curr], device=device)
            time_next_tensor = torch.tensor([time_next], device=device)

            optimizer.zero_grad()

            with torch.no_grad():
                x_base = emb_layer.get_all_entity_embeddings(time_curr_tensor).squeeze(0)
                x_spatial = spatial_layer(x_base, edge_curr, type_curr, emb_layer.rel_emb)
                H_curr, states, prev_xs = temporal_layer(x_spatial, states, prev_xs)

                subjects, objects, relations = edge_next[0], edge_next[1], type_next
                subj_states = H_curr[subjects]
                rel_embeddings = emb_layer.rel_emb(relations)
                t_emb = emb_layer.time_emb(time_next_tensor).squeeze(0).expand(len(subjects), -1)

                s_T = F.normalize(states[-1][subjects], dim=-1)
                h_c, _, _ = decomposer(subj_states, rel_embeddings)
                
                m = torch.randint(0, DIFFUSION_STEPS, (len(subjects),), device=device)
                hc_noisy, _ = scheduler.forward_noise(h_c, m, "causal")

            # --- DYNAMIC CALIBRATION ---
            hc_pred = denoiser(hc_noisy, rel_embeddings, s_T, t_emb, m)
            raw_scores = predictor(hc_pred, rel_embeddings, H_curr)
            
            # Predict how much to trust the frequency bias
            gate = confidence_head(hc_pred, rel_embeddings) # [Batch, 1]
            
            # Mix the scores dynamically
            batch_freq = freq_bias.unsqueeze(0).expand(len(subjects), -1)
            final_scores = raw_scores + gate * (LAMBDA_FREQ * batch_freq)

            loss = loss_fn(final_scores, objects)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            states, prev_xs = [s.detach() for s in states], [x.detach() for x in prev_xs]
            pbar.set_postfix({"Loss": f"{loss.item():.3f}", "Avg Gate": f"{gate.mean().item():.2f}"})

    print("\nSaving CALIBRATED Phase 2 model...")
    # Safe save to a NEW file
    torch.save({
        'denoiser': denoiser.state_dict(),
        'confidence_head': confidence_head.state_dict()
    }, "checkpoints/cdssm_calibrated.pt")
    print("✅ Saved to checkpoints/cdssm_calibrated.pt")

if __name__ == "__main__":
    train_calibrated()