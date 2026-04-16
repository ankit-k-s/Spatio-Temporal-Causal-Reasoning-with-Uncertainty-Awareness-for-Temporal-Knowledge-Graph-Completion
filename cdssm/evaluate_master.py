import torch
from collections import defaultdict
from tqdm import tqdm

from data.loader import TKGDataloader
from models.embedding import TKGEmbedding
from models.spatial import GraphMambaLayer
from models.temporal import GraphSSM
from models.causal import CausalDecomposer
from models.predictor import TKGCFusingPredictor
from models.diffusion import AsymmetricNoiseScheduler, BiSSMDenoiser

def build_filter_dict(loader):
    filter_dict = defaultdict(set)
    for split in [loader.train_snapshots, loader.valid_snapshots, loader.test_snapshots]:
        for tau, (edge_index, edge_type, _) in split.items():
            heads = edge_index[0].tolist()
            tails = edge_index[1].tolist()
            rels = edge_type.tolist()
            for h, r, t in zip(heads, rels, tails):
                filter_dict[(h, r, tau)].add(t)
    return filter_dict

def evaluate_master():
    print("\n" + "="*60)
    print("CD-SSM TRACK 2: MASTER HYBRID EVALUATOR")
    print("="*60)

    # --- SOTA HYPERPARAMS ---
    K_SAMPLES = 5
    GUIDANCE_STEP = 20
    ALPHA = 0.5  # 50% Det / 50% Diff
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = TKGDataloader(data_dir="data/ICEWS14")
    filter_dict = build_filter_dict(loader)
    
    # --- TRACK 2 ARCHITECTURE ---
    N = loader.num_entities
    D = 384
    D_STATE = 16 
    DIFFUSION_STEPS = 100
    DENOISER_LAYERS = 3

    emb_layer = TKGEmbedding(N, loader.num_relations_total, len(loader.time2id), D).to(device)
    spatial_layer = GraphMambaLayer(d_model=D, seq_len=3).to(device)
    temporal_layer = GraphSSM(d_model=D, d_state=D_STATE, num_layers=2).to(device)
    decomposer = CausalDecomposer(d_model=D).to(device)
    predictor = TKGCFusingPredictor(d_model=D).to(device)
    scheduler = AsymmetricNoiseScheduler(num_timesteps=DIFFUSION_STEPS).to(device)
    denoiser = BiSSMDenoiser(d_model=D, d_state=D_STATE, num_timesteps=DIFFUSION_STEPS, num_layers=DENOISER_LAYERS).to(device)

    # Load Weights
    checkpoint = torch.load("checkpoints/cdssm_phase2.pt", map_location=device, weights_only=True)
    emb_layer.load_state_dict(checkpoint['emb_layer'])
    spatial_layer.load_state_dict(checkpoint['spatial_layer'])
    temporal_layer.load_state_dict(checkpoint['temporal_layer'])
    decomposer.load_state_dict(checkpoint['decomposer'])
    predictor.load_state_dict(checkpoint['predictor'])
    denoiser.load_state_dict(checkpoint['denoiser'])

    for m in [emb_layer, spatial_layer, temporal_layer, decomposer, predictor, denoiser]: m.eval()

    with torch.no_grad():
        print("[1] Rolling Temporal Memory (Warmup)...")
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

        print("\n[2] RUNNING HYBRID EVALUATION (DET vs DIFF vs MERGE)")
        
        mrr_det, mrr_diff, mrr_hybrid = 0.0, 0.0, 0.0
        total_queries = 0
        
        test_times = sorted(loader.test_snapshots.keys())
        for tau in tqdm(test_times, desc="Testing"):
            edge_index, edge_type, time_id = loader.test_snapshots[tau]
            edge_index, edge_type = edge_index.to(device), edge_type.to(device)
            time_tensor = torch.tensor([time_id], dtype=torch.long, device=device)
            
            subjects = edge_index[0]
            objects = edge_index[1]
            relations = edge_type
            batch_size = len(subjects)
            
            # --- 1. DETERMINISTIC SCORING ---
            subj_states = H_prev[subjects]
            rel_embeddings = emb_layer.rel_emb(relations)
            h_c, h_s, _ = decomposer(subj_states, rel_embeddings)
            h_do = h_c + decomposer.intervene(h_s)
            scores_det = predictor(h_do, rel_embeddings, H_prev) 

            # --- 2. DIFFUSION ENSEMBLE SCORING ---
            h_c_k = h_c.repeat_interleave(K_SAMPLES, dim=0)
            rel_emb_k = rel_embeddings.repeat_interleave(K_SAMPLES, dim=0)
            t_emb_k = emb_layer.time_emb(time_tensor).squeeze(0).expand(batch_size, -1).repeat_interleave(K_SAMPLES, dim=0)
            s_T_raw_k = states[-1][subjects].repeat_interleave(K_SAMPLES, dim=0)
            
            m_guide_k = torch.full((batch_size * K_SAMPLES,), GUIDANCE_STEP, dtype=torch.long, device=device)
            hc_noisy_k, _ = scheduler.forward_noise(h_c_k, m_guide_k, branch_type="causal")
            
            hc_pred_k = denoiser(hc_noisy_k, rel_emb_k, s_T_raw_k, t_emb_k, m_guide_k)
            # NaN Safety
            hc_pred_k = torch.nan_to_num(hc_pred_k, nan=0.0, posinf=1.0, neginf=-1.0)
            
            scores_diff_k = predictor(hc_pred_k, rel_emb_k, H_prev)
            scores_diff_k = scores_diff_k.reshape(batch_size, K_SAMPLES, N)
            
            # Use Top-K pooling to eliminate bad noise outliers
            scores_diff = scores_diff_k.topk(2, dim=1).values.mean(dim=1)

            # --- 3. HYBRID MERGE ---
            scores_hybrid = (ALPHA * scores_det) + ((1.0 - ALPHA) * scores_diff)

            # --- FILTERING & RANKING ---
            for i in range(batch_size):
                h, r, true_tail = subjects[i].item(), relations[i].item(), objects[i].item()
                valid_tails = filter_dict[(h, r, tau)]
                mask_idx = [t for t in valid_tails if t != true_tail]
                if mask_idx:
                    scores_det[i, mask_idx] = -1e6
                    scores_diff[i, mask_idx] = -1e6
                    scores_hybrid[i, mask_idx] = -1e6
            
            ranks_det = (scores_det.argsort(dim=-1, descending=True) == objects.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1.0
            ranks_diff = (scores_diff.argsort(dim=-1, descending=True) == objects.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1.0
            ranks_hybrid = (scores_hybrid.argsort(dim=-1, descending=True) == objects.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1.0
            
            mrr_det += (1.0 / ranks_det).sum().item()
            mrr_diff += (1.0 / ranks_diff).sum().item()
            mrr_hybrid += (1.0 / ranks_hybrid).sum().item()
            total_queries += batch_size

            # Update Memory
            x_t_base = emb_layer.get_all_entity_embeddings(time_tensor).squeeze(0)
            x_t_spatial = spatial_layer(x_t_base, edge_index, edge_type, emb_layer.rel_emb)
            H_prev, states, prev_xs = temporal_layer(x_t_spatial, states, prev_xs)

        print("\n" + "="*40)
        print("====== TRACK 2 MASTER DIAGNOSTIC ======")
        print(f"Phase 1 Deterministic MRR : {mrr_det / total_queries:.4f}")
        print(f"Phase 2 Pure Diffusion MRR: {mrr_diff / total_queries:.4f}")
        print(f"Hybrid Final MRR (a=0.5)  : {mrr_hybrid / total_queries:.4f}")
        print("="*40 + "\n")

if __name__ == "__main__":
    evaluate_master()