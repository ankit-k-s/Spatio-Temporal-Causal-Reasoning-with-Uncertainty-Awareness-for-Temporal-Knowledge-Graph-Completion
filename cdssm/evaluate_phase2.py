import torch
from collections import defaultdict
from tqdm import tqdm

from data.loader import TKGDataloader
from models.embedding import TKGEmbedding
from models.spatial import GraphMambaLayer
from models.temporal import GraphSSM
from models.causal import CausalDecomposer
from models.predictor import TKGCFusingPredictor
from models.diffusion import BiSSMDenoiser

def build_filter_dict(loader):
    filter_dict = defaultdict(set)
    for split in [loader.train_snapshots, loader.valid_snapshots, loader.test_snapshots]:
        for tau, (edge_index, edge_type, _) in split.items():
            heads, tails, rels = edge_index[0].tolist(), edge_index[1].tolist(), edge_type.tolist()
            for h, r, t in zip(heads, rels, tails):
                filter_dict[(h, r, tau)].add(t)
    return filter_dict

def evaluate_phase2_ensemble():
    print("\n" + "="*60)
    print("CD-SSM PHASE 2: ENSEMBLE DIFFUSION INFERENCE")
    print("="*60)

    # --- ENSEMBLE HYPERPARAMETER ---
    K_SAMPLES = 5  # Number of futures to sample per query
    print(f"Ensemble Size: K = {K_SAMPLES} parallel futures per query\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = TKGDataloader(data_dir="data/ICEWS14")
    filter_dict = build_filter_dict(loader)
    
    N, D, D_STATE, DIFFUSION_STEPS = loader.num_entities, 200, 16, 100

    emb_layer = TKGEmbedding(N, loader.num_relations_total, len(loader.time2id), D).to(device)
    spatial_layer = GraphMambaLayer(d_model=D, seq_len=3).to(device)
    temporal_layer = GraphSSM(d_model=D, d_state=D_STATE, num_layers=2).to(device)
    decomposer = CausalDecomposer(d_model=D).to(device)
    predictor = TKGCFusingPredictor(d_model=D).to(device)
    denoiser = BiSSMDenoiser(d_model=D, d_state=D_STATE, num_timesteps=DIFFUSION_STEPS).to(device)

    # Load Phase 2 Weights
    try:
        checkpoint = torch.load("checkpoints/cdssm_phase2.pt", map_location=device, weights_only=True)
        emb_layer.load_state_dict(checkpoint['emb_layer'])
        spatial_layer.load_state_dict(checkpoint['spatial_layer'])
        temporal_layer.load_state_dict(checkpoint['temporal_layer'])
        decomposer.load_state_dict(checkpoint['decomposer'])
        predictor.load_state_dict(checkpoint['predictor'])
        denoiser.load_state_dict(checkpoint['denoiser'])
    except FileNotFoundError:
        print(" Error: checkpoints/cdssm_phase2.pt not found!")
        return

    for m in [emb_layer, spatial_layer, temporal_layer, decomposer, predictor, denoiser]: m.eval()

    with torch.no_grad():
        print("[1] Rolling Temporal Memory (Train + Valid)...")
        states, prev_xs = None, None
        
        warmup_times = sorted(list(loader.train_snapshots.keys()) + list(loader.valid_snapshots.keys()))
        all_snapshots = {**loader.train_snapshots, **loader.valid_snapshots, **loader.test_snapshots}
        
        H_prev = torch.zeros(N, D).to(device)
        
        for tau in warmup_times:
            edge_index, edge_type, time_id = all_snapshots[tau]
            edge_index, edge_type = edge_index.to(device), edge_type.to(device)
            time_tensor = torch.tensor([time_id], dtype=torch.long, device=device)
            
            x_t_base = emb_layer.get_all_entity_embeddings(time_tensor).squeeze(0)
            x_t_spatial = spatial_layer(x_t_base, edge_index, edge_type, emb_layer.rel_emb)
            H_prev, states, prev_xs = temporal_layer(x_t_spatial, states, prev_xs)

        print("\n[2] GENERATIVE ENSEMBLE EVALUATION: TEST SET")
        mrr, hits_1, hits_10, total_queries = 0.0, 0.0, 0.0, 0
        
        test_times = sorted(loader.test_snapshots.keys())
        for tau in tqdm(test_times, desc="Testing"):
            edge_index, edge_type, time_id = loader.test_snapshots[tau]
            edge_index, edge_type = edge_index.to(device), edge_type.to(device)
            time_tensor = torch.tensor([time_id], dtype=torch.long, device=device)
            
            subjects, objects, relations = edge_index[0], edge_index[1], edge_type
            batch_size = len(subjects)
            
            # --- THE ENSEMBLE SAMPLER (OPTION A) ---
            
            # 1. Expand the conditions K times
            rel_emb = emb_layer.rel_emb(relations)
            rel_emb_k = rel_emb.repeat_interleave(K_SAMPLES, dim=0) # [Batch * K, D]
            
            t_emb = emb_layer.time_emb(time_tensor).squeeze(0).expand(batch_size, -1)
            t_emb_k = t_emb.repeat_interleave(K_SAMPLES, dim=0) # [Batch * K, D]
            
            s_T_raw = states[-1][subjects]
            s_T_raw_k = s_T_raw.repeat_interleave(K_SAMPLES, dim=0) # [Batch * K, D, D_STATE]
            
            # 2. Set diffusion step to max (T) for pure noise
            m_max_k = torch.full((batch_size * K_SAMPLES,), DIFFUSION_STEPS - 1, dtype=torch.long, device=device)
            
            # 3. Generate K DIFFERENT noise vectors for every single query
            pure_noise_k = torch.randn(batch_size * K_SAMPLES, D, device=device)
            
            # 4. Denoise all futures simultaneously
            hc_pred_k = denoiser(pure_noise_k, rel_emb_k, s_T_raw_k, t_emb_k, m_max_k)
            
            # 5. Score all futures
            scores_k = predictor(hc_pred_k, rel_emb_k, H_prev) # [Batch * K, N]
            
            # 6. Aggregate: Reshape and average the probabilities to find the consensus!
            # Shape goes from [Batch * K, N] -> [Batch, K, N] -> mean -> [Batch, N]
            scores_ensembled = scores_k.view(batch_size, K_SAMPLES, N).mean(dim=1)

            # --- FILTERED EVALUATION PROTOCOL ---
            for i in range(batch_size):
                h, r, true_tail = subjects[i].item(), relations[i].item(), objects[i].item()
                valid_tails = filter_dict[(h, r, tau)]
                mask_idx = [t for t in valid_tails if t != true_tail]
                if mask_idx:
                    scores_ensembled[i, mask_idx] = -1e9
            
            sorted_indices = scores_ensembled.argsort(dim=-1, descending=True)
            ranks = (sorted_indices == objects.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1.0
            
            mrr += (1.0 / ranks).sum().item()
            hits_1 += (ranks <= 1).sum().item()
            hits_10 += (ranks <= 10).sum().item()
            total_queries += batch_size

            # --- UPDATE MEMORY FOR NEXT TIMESTEP ---
            x_t_base = emb_layer.get_all_entity_embeddings(time_tensor).squeeze(0)
            x_t_spatial = spatial_layer(x_t_base, edge_index, edge_type, emb_layer.rel_emb)
            H_prev, states, prev_xs = temporal_layer(x_t_spatial, states, prev_xs)

        print("\n" + "="*40)
        print("====== PHASE 2 ENSEMBLE RESULTS ======")
        print("="*40)
        print(f"Phase 1 Baseline MRR : 0.2227")
        print(f"Phase 2 Diffused MRR : {mrr / total_queries:.4f}")
        print(f"Hits@1               : {hits_1 / total_queries:.4f}")
        print(f"Hits@10              : {hits_10 / total_queries:.4f}")
        print("="*40 + "\n")

if __name__ == "__main__":
    evaluate_phase2_ensemble()