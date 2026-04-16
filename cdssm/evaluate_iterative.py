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

def evaluate_iterative_diffusion():
    print("\n" + "="*60)
    print("CD-SSM ULTIMATE EVALUATION: ITERATIVE REVERSE DDPM")
    print("="*60)

    # --- BEST CONFIG FROM YOUR SWEEP ---
    K_SAMPLES = 5         
    GUIDANCE_STEP = 20    
    
    print(f"Ensemble Size : K = {K_SAMPLES}")
    print(f"Guidance Step : m = {GUIDANCE_STEP} (Stepping down iteratively to 0)\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = TKGDataloader(data_dir="data/ICEWS14")
    filter_dict = build_filter_dict(loader)
    
    N, D, D_STATE, DIFFUSION_STEPS = loader.num_entities, 200, 16, 100

    emb_layer = TKGEmbedding(N, loader.num_relations_total, len(loader.time2id), D).to(device)
    spatial_layer = GraphMambaLayer(d_model=D, seq_len=3).to(device)
    temporal_layer = GraphSSM(d_model=D, d_state=D_STATE, num_layers=2).to(device)
    decomposer = CausalDecomposer(d_model=D).to(device)
    predictor = TKGCFusingPredictor(d_model=D).to(device)
    scheduler = AsymmetricNoiseScheduler(num_timesteps=DIFFUSION_STEPS, elevated_beta=5.0).to(device)
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

        print("\n[2] ITERATIVE REVERSE SAMPLING: TEST SET")
        mrr, hits_1, hits_10, total_queries = 0.0, 0.0, 0.0, 0
        
        test_times = sorted(loader.test_snapshots.keys())
        for tau in tqdm(test_times, desc="Testing"):
            edge_index, edge_type, time_id = loader.test_snapshots[tau]
            edge_index, edge_type = edge_index.to(device), edge_type.to(device)
            time_tensor = torch.tensor([time_id], dtype=torch.long, device=device)
            
            subjects, objects, relations = edge_index[0], edge_index[1], edge_type
            batch_size = len(subjects)
            
            # --- EXTRACT DETERMINISTIC PRIOR & K-EXPAND ---
            subj_states = H_prev[subjects]
            rel_embeddings = emb_layer.rel_emb(relations)
            h_c, _, _ = decomposer(subj_states, rel_embeddings)
            
            h_c_k = h_c.repeat_interleave(K_SAMPLES, dim=0) 
            rel_emb_k = rel_embeddings.repeat_interleave(K_SAMPLES, dim=0)
            t_emb_k = emb_layer.time_emb(time_tensor).squeeze(0).expand(batch_size, -1).repeat_interleave(K_SAMPLES, dim=0)
            s_T_raw_k = states[-1][subjects].repeat_interleave(K_SAMPLES, dim=0)
            
            # --- START NOISE ANCHOR (m = GUIDANCE_STEP) ---
            m_guide_k = torch.full((batch_size * K_SAMPLES,), GUIDANCE_STEP, dtype=torch.long, device=device)
            x_t, _ = scheduler.forward_noise(h_c_k, m_guide_k, branch_type="causal")
            
            # --- THE ITERATIVE REVERSE DIFFUSION CHAIN (The SOTA Fix) ---
            for step in reversed(range(GUIDANCE_STEP + 1)):
                t_tensor = torch.full((batch_size * K_SAMPLES,), step, dtype=torch.long, device=device)
                
                # 1. Denoiser predicts the clean state x_0 from the current noisy state x_t
                x_0_pred = denoiser(x_t, rel_emb_k, s_T_raw_k, t_emb_k, t_tensor)
                
                # 2. Step down mathematically to x_{t-1}
                if step > 0:
                    x_t = scheduler.reverse_step(x_t, x_0_pred, t_tensor)
                else:
                    # At step 0, our final answer is the prediction itself
                    hc_final_k = x_0_pred 
            
            # --- SCORE & AGGREGATE ---
            scores_k = predictor(hc_final_k, rel_emb_k, H_prev) 
            scores_ensembled = scores_k.view(batch_size, K_SAMPLES, N).mean(dim=1)

            # --- FILTERED PROTOCOL ---
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

            # Update Memory
            x_t_base = emb_layer.get_all_entity_embeddings(time_tensor).squeeze(0)
            x_t_spatial = spatial_layer(x_t_base, edge_index, edge_type, emb_layer.rel_emb)
            H_prev, states, prev_xs = temporal_layer(x_t_spatial, states, prev_xs)

        print("\n" + "="*40)
        print("====== ITERATIVE SOTA RESULTS ======")
        print("="*40)
        print(f"Target to Beat (DiffuTKG) : 0.4850")
        print(f"CD-SSM Iterative MRR      : {mrr / total_queries:.4f}")
        print(f"Hits@1                    : {hits_1 / total_queries:.4f}")
        print(f"Hits@10                   : {hits_10 / total_queries:.4f}")
        print("="*40 + "\n")

if __name__ == "__main__":
    evaluate_iterative_diffusion()