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
from models.calibration import ConfidenceHead

def build_filter_dict(loader):
    filter_dict = defaultdict(set)
    for split in [loader.train_snapshots, loader.valid_snapshots, loader.test_snapshots]:
        for tau, (edge_index, edge_type, _) in split.items():
            heads, tails = edge_index[0].tolist(), edge_index[1].tolist()
            rels = edge_type.tolist()
            for h, r, t in zip(heads, rels, tails): filter_dict[(h, r, tau)].add(t)
    return filter_dict

def compute_frequency_prior(loader, N, device):
    entity_freq = torch.zeros(N, device=device)
    for tau, (edge_index, _, _) in loader.train_snapshots.items():
        objects = edge_index[1].to(device)
        entity_freq.scatter_add_(0, objects, torch.ones_like(objects, dtype=torch.float))
    return torch.log(entity_freq / entity_freq.sum() + 1e-9)

def evaluate_calibrated():
    print("\n" + "="*60)
    print("CD-SSM SOTA INFERENCE: TRAINED UNCERTAINTY GATING")
    print("="*60)

    K_SAMPLES, GUIDANCE_STEP, ALPHA, TAU, LAMBDA_FREQ = 5, 20, 0.5, 0.5, 0.15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = TKGDataloader(data_dir="data/ICEWS14")
    filter_dict = build_filter_dict(loader)
    
    N, D, D_STATE, DIFF_STEPS = loader.num_entities, 200, 16, 100
    freq_bias = compute_frequency_prior(loader, N, device)

    emb_layer = TKGEmbedding(N, loader.num_relations_total, len(loader.time2id), D).to(device)
    spatial_layer = GraphMambaLayer(d_model=D, seq_len=3).to(device)
    temporal_layer = GraphSSM(d_model=D, d_state=D_STATE, num_layers=2).to(device)
    decomposer = CausalDecomposer(d_model=D).to(device)
    predictor = TKGCFusingPredictor(d_model=D).to(device)
    scheduler = AsymmetricNoiseScheduler(num_timesteps=DIFF_STEPS).to(device)
    denoiser = BiSSMDenoiser(d_model=D, d_state=D_STATE, num_timesteps=DIFF_STEPS).to(device)
    confidence_head = ConfidenceHead(d_model=D).to(device)

    # Load Weights (SAFE)
    cp1 = torch.load("checkpoints/cdssm_phase1.pt", map_location=device, weights_only=True)
    emb_layer.load_state_dict(cp1['emb_layer'])
    spatial_layer.load_state_dict(cp1['spatial_layer'])
    temporal_layer.load_state_dict(cp1['temporal_layer'])
    decomposer.load_state_dict(cp1['decomposer'])
    predictor.load_state_dict(cp1['predictor'])
    
    cp_calib = torch.load("checkpoints/cdssm_calibrated.pt", map_location=device, weights_only=True)
    denoiser.load_state_dict(cp_calib['denoiser'])
    confidence_head.load_state_dict(cp_calib['confidence_head'])

    for m in [emb_layer, spatial_layer, temporal_layer, decomposer, predictor, denoiser, confidence_head]: m.eval()

    with torch.no_grad():
        print("[1] Rolling Temporal Memory...")
        states, prev_xs = None, None
        all_snapshots = {**loader.train_snapshots, **loader.valid_snapshots, **loader.test_snapshots}
        H_prev = torch.zeros(N, D).to(device)
        
        for tau in sorted(list(loader.train_snapshots.keys()) + list(loader.valid_snapshots.keys())):
            edge_index, edge_type, time_id = all_snapshots[tau]
            edge_index, edge_type = edge_index.to(device), edge_type.to(device)
            x_t_base = emb_layer.get_all_entity_embeddings(torch.tensor([time_id], device=device)).squeeze(0)
            H_prev, states, prev_xs = temporal_layer(spatial_layer(x_t_base, edge_index, edge_type, emb_layer.rel_emb), states, prev_xs)

        print("\n[2] RUNNING CALIBRATED HYBRID EVALUATION")
        mrr, h1, h10, total_queries = 0.0, 0.0, 0.0, 0
        
        for tau in tqdm(sorted(loader.test_snapshots.keys()), desc="Testing"):
            edge_index, edge_type, time_id = loader.test_snapshots[tau]
            edge_index, edge_type = edge_index.to(device), edge_type.to(device)
            time_tensor = torch.tensor([time_id], device=device)
            
            subjects, objects, relations = edge_index[0], edge_index[1], edge_type
            batch_size = len(subjects)
            
            # Det
            h_c, h_s, _ = decomposer(H_prev[subjects], emb_layer.rel_emb(relations))
            scores_det = predictor(h_c + decomposer.intervene(h_s), emb_layer.rel_emb(relations), H_prev) / TAU

            # Diff
            h_c_k = h_c.repeat_interleave(K_SAMPLES, dim=0)
            rel_emb_k = emb_layer.rel_emb(relations).repeat_interleave(K_SAMPLES, dim=0)
            t_emb_k = emb_layer.time_emb(time_tensor).squeeze(0).expand(batch_size, -1).repeat_interleave(K_SAMPLES, dim=0)
            s_T_k = states[-1][subjects].repeat_interleave(K_SAMPLES, dim=0)
            m_k = torch.full((batch_size * K_SAMPLES,), GUIDANCE_STEP, dtype=torch.long, device=device)
            
            hc_noisy_k, _ = scheduler.forward_noise(h_c_k, m_k, "causal")
            hc_pred_k = torch.nan_to_num(denoiser(hc_noisy_k, rel_emb_k, s_T_k, t_emb_k, m_k), nan=0.0)
            
            # Hybrid Base
            scores_diff = predictor(hc_pred_k, rel_emb_k, H_prev).reshape(batch_size, K_SAMPLES, N).topk(2, dim=1).values.mean(dim=1) / TAU
            scores_hybrid = (ALPHA * scores_det) + ((1.0 - ALPHA) * scores_diff)

            # --- THE TRAINED CALIBRATION ---
            # Model predicts its uncertainty [Batch, 1]
            gate = confidence_head(hc_pred_k.reshape(batch_size, K_SAMPLES, D).mean(dim=1), emb_layer.rel_emb(relations))
            
            # Applies frequency prior ONLY when uncertain
            scores_hybrid = scores_hybrid + gate * (LAMBDA_FREQ * freq_bias.unsqueeze(0))

            for i in range(batch_size):
                h, r, true_tail = subjects[i].item(), relations[i].item(), objects[i].item()
                mask_idx = [t for t in filter_dict[(h, r, tau)] if t != true_tail]
                if mask_idx: scores_hybrid[i, mask_idx] = -1e6
            
            ranks = (scores_hybrid.argsort(dim=-1, descending=True) == objects.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1.0
            mrr += (1.0 / ranks).sum().item(); h1 += (ranks <= 1).sum().item(); h10 += (ranks <= 10).sum().item()
            total_queries += batch_size

            x_t_base = emb_layer.get_all_entity_embeddings(time_tensor).squeeze(0)
            H_prev, states, prev_xs = temporal_layer(spatial_layer(x_t_base, edge_index, edge_type, emb_layer.rel_emb), states, prev_xs)

        print("\n" + "="*40)
        print("====== TRAINED CALIBRATION SOTA ======")
        print(f"Hybrid MRR: {mrr / total_queries:.4f}")
        print(f"Hits@1    : {h1 / total_queries:.4f}")
        print(f"Hits@10   : {h10 / total_queries:.4f}")
        print("="*40 + "\n")

if __name__ == "__main__":
    evaluate_calibrated()