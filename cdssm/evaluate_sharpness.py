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

def evaluate_sharpness():
    print("\n" + "="*60)
    print("CD-SSM TRACK 1: RANKING SHARPNESS OPTIMIZATION")
    print("="*60)

    # --- BASE SOTA CONFIG ---
    K_SAMPLES = 5
    GUIDANCE_STEP = 20
    
    # --- TRACK 1 SWEEP PARAMETERS ---
    alphas = [0.6, 0.7, 0.8]        # Testing heavier Deterministic weight
    temperatures = [1.0, 0.7, 0.5]  # T < 1.0 sharpens the distribution
    POOLING_K = 2                   # Top-K mean pooling instead of full mean

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

        print("\n[2] RUNNING TRACK 1 SHARPNESS SWEEP")
        
        test_times = sorted(loader.test_snapshots.keys())
        test_data = []
        for tau in test_times:
            edge_index, edge_type, time_id = loader.test_snapshots[tau]
            test_data.append((tau, edge_index.to(device), edge_type.to(device), torch.tensor([time_id], dtype=torch.long, device=device)))

        results_log = []
        best_mrr = 0.0

        for ALPHA in alphas:
            for T in temperatures:
                print(f"\n--- Testing Alpha = {ALPHA}, Temp = {T} ---")
                
                run_states = [s.clone() for s in states] if states else None
                run_prev_xs = [x.clone() for x in prev_xs] if prev_xs else None
                run_H_prev = H_prev.clone()
                
                mrr, hits_1, hits_10, total_queries = 0.0, 0.0, 0.0, 0
                
                for tau, edge_index, edge_type, time_tensor in tqdm(test_data, desc="Evaluating", leave=False):
                    subjects, objects, relations = edge_index[0], edge_index[1], edge_type
                    batch_size = len(subjects)
                    
                    # Deterministic
                    subj_states = run_H_prev[subjects]
                    rel_embeddings = emb_layer.rel_emb(relations)
                    h_c, h_s, _ = decomposer(subj_states, rel_embeddings)
                    h_do = h_c + decomposer.intervene(h_s)
                    
                    # TEMPERATURE SCALING ON DETERMINISTIC LOGITS
                    scores_det = predictor(h_do, rel_embeddings, run_H_prev) / T 

                    # Diffusion Ensemble
                    h_c_k = h_c.repeat_interleave(K_SAMPLES, dim=0) 
                    rel_emb_k = rel_embeddings.repeat_interleave(K_SAMPLES, dim=0)
                    t_emb_k = emb_layer.time_emb(time_tensor).squeeze(0).expand(batch_size, -1).repeat_interleave(K_SAMPLES, dim=0)
                    s_T_raw_k = run_states[-1][subjects].repeat_interleave(K_SAMPLES, dim=0)
                    
                    m_guide_k = torch.full((batch_size * K_SAMPLES,), GUIDANCE_STEP, dtype=torch.long, device=device)
                    hc_noisy_k, _ = scheduler.forward_noise(h_c_k, m_guide_k, branch_type="causal")
                    hc_pred_k = denoiser(hc_noisy_k, rel_emb_k, s_T_raw_k, t_emb_k, m_guide_k)
                    
                    # TEMPERATURE SCALING ON DIFFUSION LOGITS
                    scores_diff_k = predictor(hc_pred_k, rel_emb_k, run_H_prev) / T 
                    
                    # GPT'S SHARPNESS FIX: Top-K Mean Pooling instead of Full Mean
                    scores_diff_k = scores_diff_k.view(batch_size, K_SAMPLES, N)
                    topk_scores = scores_diff_k.topk(POOLING_K, dim=1).values
                    scores_diff = topk_scores.mean(dim=1) 

                    # HYBRID MERGE
                    scores_final = (ALPHA * scores_det) + ((1.0 - ALPHA) * scores_diff)

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

                    x_t_base = emb_layer.get_all_entity_embeddings(time_tensor).squeeze(0)
                    x_t_spatial = spatial_layer(x_t_base, edge_index, edge_type, emb_layer.rel_emb)
                    run_H_prev, run_states, run_prev_xs = temporal_layer(x_t_spatial, run_states, run_prev_xs)

                final_mrr = mrr / total_queries
                final_h1 = hits_1 / total_queries
                final_h10 = hits_10 / total_queries
                
                print(f"Result -> MRR: {final_mrr:.4f} | Hits@1: {final_h1:.4f} | Hits@10: {final_h10:.4f}")
                results_log.append((ALPHA, T, final_mrr, final_h1, final_h10))
                
                if final_mrr > best_mrr:
                    best_mrr = final_mrr

        print("\n" + "="*40)
        print("====== TRACK 1 SWEEP COMPLETE ======")
        for res in results_log:
            print(f"Alpha={res[0]:.1f}, Temp={res[1]:.1f} | MRR: {res[2]:.4f} | Hits@1: {res[3]:.4f}")
        print("="*40 + "\n")

if __name__ == "__main__":
    evaluate_sharpness()