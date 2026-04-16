import torch
import copy
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
            heads, tails = edge_index[0].tolist(), edge_index[1].tolist()
            rels = edge_type.tolist()
            for h, r, t in zip(heads, rels, tails):
                filter_dict[(h, r, tau)].add(t)
    return filter_dict

def compute_frequency_prior(loader, N, device):
    """Calculates the historical popularity of entities as objects to bias Hits@1."""
    print("[*] Computing Historical Frequency Prior...")
    entity_freq = torch.zeros(N, device=device)
    for tau, (edge_index, _, _) in loader.train_snapshots.items():
        objects = edge_index[1].to(device)
        entity_freq.scatter_add_(0, objects, torch.ones_like(objects, dtype=torch.float))
    
    # Normalize and convert to log probabilities
    entity_freq = entity_freq / entity_freq.sum()
    freq_bias = torch.log(entity_freq + 1e-9) # 1e-9 prevents log(0)
    return freq_bias

def evaluate_master_sota():
    print("\n" + "="*60)
    print("CD-SSM SOTA INFERENCE: TEMPERATURE + FREQUENCY PRIOR")
    print("="*60)

    # --- THE SOTA SHARPENING HYPERPARAMS ---
    K_SAMPLES = 5
    GUIDANCE_STEP = 20
    ALPHA = 0.5         # 50% Det / 50% Diff
    TAU = 0.5           # Temperature Scaling (sharpens peaks)
    LAMBDA_FREQ = 0.15  # Weight of the Frequency Prior Bias
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = TKGDataloader(data_dir="data/ICEWS14")
    filter_dict = build_filter_dict(loader)
    
    N = loader.num_entities
    D, D_STATE, DIFFUSION_STEPS = 200, 16, 100
    
    freq_bias = compute_frequency_prior(loader, N, device)

    emb_layer = TKGEmbedding(N, loader.num_relations_total, len(loader.time2id), D).to(device)
    spatial_layer = GraphMambaLayer(d_model=D, seq_len=3).to(device)
    temporal_layer = GraphSSM(d_model=D, d_state=D_STATE, num_layers=2).to(device)
    decomposer = CausalDecomposer(d_model=D).to(device)
    predictor = TKGCFusingPredictor(d_model=D).to(device)
    scheduler = AsymmetricNoiseScheduler(num_timesteps=DIFFUSION_STEPS).to(device)
    denoiser = BiSSMDenoiser(d_model=D, d_state=D_STATE, num_timesteps=DIFFUSION_STEPS).to(device)

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

        print("\n[2] RUNNING HYBRID EVALUATION")
        mrr_hybrid, hits_1, hits_10 = 0.0, 0.0, 0.0
        total_queries = 0
        
        test_times = sorted(loader.test_snapshots.keys())
        for tau in tqdm(test_times, desc="Testing"):
            edge_index, edge_type, time_id = loader.test_snapshots[tau]
            edge_index, edge_type = edge_index.to(device), edge_type.to(device)
            time_tensor = torch.tensor([time_id], dtype=torch.long, device=device)
            
            subjects, objects, relations = edge_index[0], edge_index[1], edge_type
            batch_size = len(subjects)
            
            # 1. Deterministic
            subj_states = H_prev[subjects]
            rel_embeddings = emb_layer.rel_emb(relations)
            h_c, h_s, _ = decomposer(subj_states, rel_embeddings)
            h_do = h_c + decomposer.intervene(h_s)
            
            # --- APPLY TEMPERATURE SCALING ---
            scores_det = predictor(h_do, rel_embeddings, H_prev) / TAU

            # 2. Diffusion
            h_c_k = h_c.repeat_interleave(K_SAMPLES, dim=0)
            rel_emb_k = rel_embeddings.repeat_interleave(K_SAMPLES, dim=0)
            t_emb_k = emb_layer.time_emb(time_tensor).squeeze(0).expand(batch_size, -1).repeat_interleave(K_SAMPLES, dim=0)
            s_T_raw_k = states[-1][subjects].repeat_interleave(K_SAMPLES, dim=0)
            m_guide_k = torch.full((batch_size * K_SAMPLES,), GUIDANCE_STEP, dtype=torch.long, device=device)
            
            hc_noisy_k, _ = scheduler.forward_noise(h_c_k, m_guide_k, branch_type="causal")
            hc_pred_k = denoiser(hc_noisy_k, rel_emb_k, s_T_raw_k, t_emb_k, m_guide_k)
            hc_pred_k = torch.nan_to_num(hc_pred_k, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # --- APPLY TEMPERATURE SCALING ---
            scores_diff_k = predictor(hc_pred_k, rel_emb_k, H_prev) / TAU
            scores_diff_k = scores_diff_k.reshape(batch_size, K_SAMPLES, N)
            scores_diff = scores_diff_k.topk(2, dim=1).values.mean(dim=1)

            # 3. Hybrid Merge
            scores_hybrid = (ALPHA * scores_det) + ((1.0 - ALPHA) * scores_diff)

            # --- APPLY FREQUENCY PRIOR BIAS ---
            scores_hybrid = scores_hybrid + (LAMBDA_FREQ * freq_bias.unsqueeze(0))

            # Filter
            for i in range(batch_size):
                h, r, true_tail = subjects[i].item(), relations[i].item(), objects[i].item()
                valid_tails = filter_dict[(h, r, tau)]
                mask_idx = [t for t in valid_tails if t != true_tail]
                if mask_idx:
                    scores_hybrid[i, mask_idx] = -1e6
            
            ranks = (scores_hybrid.argsort(dim=-1, descending=True) == objects.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1.0
            
            mrr_hybrid += (1.0 / ranks).sum().item()
            hits_1 += (ranks <= 1).sum().item()
            hits_10 += (ranks <= 10).sum().item()
            total_queries += batch_size

            x_t_base = emb_layer.get_all_entity_embeddings(time_tensor).squeeze(0)
            x_t_spatial = spatial_layer(x_t_base, edge_index, edge_type, emb_layer.rel_emb)
            H_prev, states, prev_xs = temporal_layer(x_t_spatial, states, prev_xs)

        print("\n" + "="*40)
        print("====== SOTA INFERENCE RESULTS ======")
        print(f"Hybrid MRR (with Priors): {mrr_hybrid / total_queries:.4f}")
        print(f"Hits@1                  : {hits_1 / total_queries:.4f}")
        print(f"Hits@10                 : {hits_10 / total_queries:.4f}")
        print("="*40 + "\n")

if __name__ == "__main__":
    evaluate_master_sota()