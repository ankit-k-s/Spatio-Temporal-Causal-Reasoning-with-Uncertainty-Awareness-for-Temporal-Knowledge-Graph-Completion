import os
import torch
from torch.optim import AdamW
from tqdm import tqdm

from data.loader import TKGDataloader
from models.embedding import TKGEmbedding
from models.spatial import GraphMambaLayer
from models.temporal import GraphSSM
from models.causal import CausalDecomposer
from models.predictor import TKGCFusingPredictor
from core.loss import QuadObjectiveLoss

def train_phase_1():
    print("Initializing CD-SSM Phase 1 Training...")
    
    loader = TKGDataloader(data_dir="data/ICEWS14")
    N = loader.num_entities
    NUM_REL = loader.num_relations_total
    NUM_TIME = len(loader.time2id)
    
    if N <= 1 or NUM_TIME == 0:
        print(" Error: Dataset failed to load. Check your 'data/ICEWS14' path.")
        return

    D = 200
    D_STATE = 16
    LR = 1e-3
    EPOCHS = 1 # Keep at 1 for CPU test, change to 100 on H100
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    emb_layer = TKGEmbedding(N, NUM_REL, NUM_TIME, D).to(device)
    spatial_layer = GraphMambaLayer(d_model=D, seq_len=3).to(device)
    temporal_layer = GraphSSM(d_model=D, d_state=D_STATE, num_layers=2).to(device)
    decomposer = CausalDecomposer(d_model=D).to(device)
    predictor = TKGCFusingPredictor(d_model=D).to(device)
    loss_fn = QuadObjectiveLoss(lambda_do=0.1, lambda_unif=0.1, lambda_mask=0.01).to(device)

    params = list(emb_layer.parameters()) + list(spatial_layer.parameters()) + \
             list(temporal_layer.parameters()) + list(decomposer.parameters()) + \
             list(predictor.parameters())
    optimizer = AdamW(params, lr=LR, weight_decay=1e-4)

    # Make checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        states, prev_xs = None, None
        epoch_loss = 0.0
        
        sorted_times = sorted(loader.train_snapshots.keys())
        pbar = tqdm(sorted_times, desc="Stepping through Time")
        
        for tau in pbar:
            edge_index, edge_type, time_id = loader.train_snapshots[tau]
            edge_index, edge_type = edge_index.to(device), edge_type.to(device)
            
            optimizer.zero_grad()
            
            x_t_base = emb_layer.get_all_entity_embeddings(time_id).squeeze(0)
            x_t_spatial = spatial_layer(x_t_base, edge_index, edge_type, emb_layer.rel_emb)
            H_t, states, prev_xs = temporal_layer(x_t_spatial, states, prev_xs)
            
            states = [s.detach() for s in states]
            prev_xs = [x.detach() for x in prev_xs]
            
            subjects = edge_index[0]
            objects = edge_index[1]
            relations = edge_type
            
            subj_states = H_t[subjects]
            rel_embeddings = emb_layer.rel_emb(relations)
            
            h_c, h_s, mask = decomposer(subj_states, rel_embeddings)
            h_do = h_c + decomposer.intervene(h_s)
            
            scores_c = predictor(h_c, rel_embeddings, H_t)
            scores_s = predictor(h_s, rel_embeddings, H_t)
            scores_do = predictor(h_do, rel_embeddings, H_t)
            
            loss, metrics = loss_fn(scores_c, scores_do, scores_s, mask, objects)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            
            pbar.set_postfix({
                'Loss': f"{loss.item():.2f}", 
                'Rank': f"{metrics['loss_rank']:.2f}",
                'M_Mean': f"{mask.mean().item():.2f}"
            })

        if len(sorted_times) > 0:
            print(f"Epoch {epoch+1} Complete. Avg Loss: {epoch_loss / len(sorted_times):.4f}")
        
    # --- SAVE THE MODEL ---
    print("\nSaving Phase 1 Checkpoint...")
    checkpoint = {
        'emb_layer': emb_layer.state_dict(),
        'spatial_layer': spatial_layer.state_dict(),
        'temporal_layer': temporal_layer.state_dict(),
        'decomposer': decomposer.state_dict(),
        'predictor': predictor.state_dict()
    }
    torch.save(checkpoint, "checkpoints/cdssm_phase1.pt")
    print(" Model saved to 'checkpoints/cdssm_phase1.pt'")

if __name__ == "__main__":
    train_phase_1()