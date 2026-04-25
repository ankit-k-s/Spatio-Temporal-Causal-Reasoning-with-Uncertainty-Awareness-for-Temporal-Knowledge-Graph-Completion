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
            heads, tails, rels = edge_index[0].tolist(), edge_index[1].tolist(), edge_type.tolist()
            for h, r, t in zip(heads, rels, tails):
                filter_dict[(h, r, tau)].add(t)
    return filter_dict

def evaluate_hybrid_sota():
    print("\n" + "="*60)
    print("CD-SSM ULTIMATE EVALUATION: HYBRID INFERENCE (DET + DIFF)")
    print("="*60)

    # --- SOTA CONFIGURATION ---
    K_SAMPLES = 5
    GUIDANCE_STEP = 20
    ALPHA = 0.50  # 50% Phase 1 Deterministic, 50% Phase 2 Probabilistic
    
    print(f"Ensemble Size : K = {K_SAMPLES}")
    print(f"Guidance Step : m = {GUIDANCE_STEP}")
    print(f"Hybrid Alpha  : {ALPHA} (Weight of Deterministic Prior)\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = TKGDataloader(data_dir="data/ICEWS14")
    filter_dict = build_filter_dict(loader)
    
    N, D, D_STATE, DIFFUSION_STEPS = loader.num_entities, 200, 16, 100

    emb_layer = TKGEmbedding(N, loader.num_relations_total, len(loader.time2id), D).to(device)
    spatial_layer = GraphMambaLayer(d_model=D, seq_len=3).to(device)
    temporal_layer = GraphSSM(d_model=D, d_state=D_STATE, num_layers=2).to(device)
    decomposer = CausalDecomposer(d_model=D).to(device)
    predictor = TKGCFusingPredictor(d_model=D).to(device)
    scheduler = AsymmetricNoiseScheduler(num_timesteps=DIFFUSION_STEPS).to(device)
    denoiser = BiSSMDenoiser(d_model=D, d_state=D_STATE, num_timesteps=DIFFUSION_STEPS).to(device)

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

        print("\n[2] RUNNING HYBRID INFERENCE: TEST SET")
        mrr, hits_1, hits_10, total_queries = 0.0, 0.0, 0.0, 0
        
        test_times = sorted(loader.test_snapshots.keys())
        for tau in tqdm(test_times, desc="Testing"):
            edge_index, edge_type, time_id = loader.test_snapshots[tau]
            edge_index, edge_type = edge_index.to(device), edge_type.to(device)
            time_tensor = torch.tensor([time_id], dtype=torch.long, device=device)
            
            subjects, objects, relations = edge_index[0], edge_index[1], edge_type
            batch_size = len(subjects)
            
            subj_states = H_prev[subjects]
            rel_embeddings = emb_layer.rel_emb(relations)
            
            # ==================================================
            #  1. THE DETERMINISTIC SCORING (Phase 1 Logic)
            # ==================================================
            h_c, h_s, _ = decomposer(subj_states, rel_embeddings)
            h_do = h_c + decomposer.intervene(h_s) 
            scores_det = predictor(h_do, rel_embeddings, H_prev) # [Batch, N]

            # ==================================================
            #  2. THE DIFFUSION ENSEMBLE SCORING (Phase 2 Logic)
            # ==================================================
            h_c_k = h_c.repeat_interleave(K_SAMPLES, dim=0) 
            rel_emb_k = rel_embeddings.repeat_interleave(K_SAMPLES, dim=0)
            t_emb_k = emb_layer.time_emb(time_tensor).squeeze(0).expand(batch_size, -1).repeat_interleave(K_SAMPLES, dim=0)
            s_T_raw_k = states[-1][subjects].repeat_interleave(K_SAMPLES, dim=0)
            
            m_guide_k = torch.full((batch_size * K_SAMPLES,), GUIDANCE_STEP, dtype=torch.long, device=device)
            hc_noisy_k, _ = scheduler.forward_noise(h_c_k, m_guide_k, branch_type="causal")
            
            hc_pred_k = denoiser(hc_noisy_k, rel_emb_k, s_T_raw_k, t_emb_k, m_guide_k)
            scores_diff_k = predictor(hc_pred_k, rel_emb_k, H_prev) 
            scores_diff = scores_diff_k.view(batch_size, K_SAMPLES, N).mean(dim=1) # [Batch, N]

            # ==================================================
            #  3. THE HYBRID MERGE
            # ==================================================
            scores_final = (ALPHA * scores_det) + ((1.0 - ALPHA) * scores_diff)

            # --- FILTERED PROTOCOL ---
            for i in range(batch_size):
                h, r, true_tail = subjects[i].item(), relations[i].item(), objects[i].item()
                valid_tails = filter_dict[(h, r, tau)]
                mask_idx = [t for t in valid_tails if t != true_tail]
                if mask_idx:
                    scores_final[i, mask_idx] = -1e9
            
            sorted_indices = scores_final.argsort(dim=-1, descending=True)
            ranks = (sorted_indices == objects.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1.0
            
            mrr += (1.0 / ranks).sum().item()
            hits_1 += (ranks <= 1).sum().item()
            hits_10 += (ranks <= 10).sum().item()
            total_queries += batch_size

            # Update Memory
            x_t_base = emb_layer.get_all_entity_embeddings(time_tensor).squeeze(0)
            x_t_spatial = spatial_layer(x_t_base, edge_index, edge_type, emb_layer.rel_emb)
            H_prev, states, prev_xs = temporal_layer(x_t_spatial, states, prev_xs)

        print("\n" + "="*40)
        print("====== HYBRID SOTA RESULTS ======")
        print("="*40)
        print(f"Target to Beat (DiffuTKG) : 0.4850")
        print(f"CD-SSM Hybrid MRR         : {mrr / total_queries:.4f}")
        print(f"Hits@1                    : {hits_1 / total_queries:.4f}")
        print(f"Hits@10                   : {hits_10 / total_queries:.4f}")
        print("="*40 + "\n")

if __name__ == "__main__":
    evaluate_hybrid_sota()