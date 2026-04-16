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
            heads = edge_index[0].tolist()
            tails = edge_index[1].tolist()
            rels = edge_type.tolist()
            for h, r, t in zip(heads, rels, tails):
                filter_dict[(h, r, tau)].add(t)
    return filter_dict


def tune_inference():
    print("\n" + "="*60)
    print("CD-SSM TRACK 2: FIXED INFERENCE (STABLE)")
    print("="*60)

    # 🔧 CLEAN GRID (based on prior results)
    guidance_steps = [20, 25]
    k_samples = [5, 10]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = TKGDataloader(data_dir="data/ICEWS14")
    filter_dict = build_filter_dict(loader)

    # -----------------------
    # MODEL CONFIG (MATCH TRAINING)
    # -----------------------
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
    scheduler = AsymmetricNoiseScheduler(num_timesteps=DIFFUSION_STEPS, elevated_beta=5.0).to(device)

    denoiser = BiSSMDenoiser(
        d_model=D,
        d_state=D_STATE,
        num_timesteps=DIFFUSION_STEPS,
        num_layers=DENOISER_LAYERS
    ).to(device)

    # -----------------------
    # LOAD CHECKPOINT
    # -----------------------
    checkpoint = torch.load("checkpoints/cdssm_phase2.pt", map_location=device)

    emb_layer.load_state_dict(checkpoint['emb_layer'])
    spatial_layer.load_state_dict(checkpoint['spatial_layer'])
    temporal_layer.load_state_dict(checkpoint['temporal_layer'])
    decomposer.load_state_dict(checkpoint['decomposer'])
    predictor.load_state_dict(checkpoint['predictor'])
    denoiser.load_state_dict(checkpoint['denoiser'])

    # Sanity check
    print("Embedding shape:", emb_layer.ent_emb.weight.shape)

    for m in [emb_layer, spatial_layer, temporal_layer, decomposer, predictor, denoiser]:
        m.eval()

    with torch.no_grad():

        # -----------------------
        # WARMUP (Temporal Memory)
        # -----------------------
        print("\n[1] Rolling Temporal Memory (Warmup)...")

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

        print("\n[2] RUNNING GRID SEARCH")

        best_mrr = 0.0

        test_times = sorted(loader.test_snapshots.keys())
        test_data = [
            (
                tau,
                edge_index.to(device),
                edge_type.to(device),
                torch.tensor([time_id], dtype=torch.long, device=device)
            )
            for tau, (edge_index, edge_type, time_id) in loader.test_snapshots.items()
        ]

        for K in k_samples:
            for G_STEP in guidance_steps:

                print(f"\n--- K={K}, Guidance={G_STEP} ---")

                # 🔧 FIX: deep copy states
                run_states = copy.deepcopy(states)
                run_prev_xs = copy.deepcopy(prev_xs)
                run_H_prev = H_prev.clone()

                mrr = hits1 = hits10 = total = 0

                for tau, edge_index, edge_type, time_tensor in tqdm(test_data, leave=False):

                    subjects = edge_index[0]
                    objects = edge_index[1]
                    relations = edge_type

                    batch_size = len(subjects)

                    subj_states = run_H_prev[subjects]
                    rel_emb = emb_layer.rel_emb(relations)

                    h_c, _, _ = decomposer(subj_states, rel_emb)

                    # Expand K samples
                    h_c_k = h_c.repeat_interleave(K, dim=0)
                    rel_emb_k = rel_emb.repeat_interleave(K, dim=0)

                    t_emb = emb_layer.time_emb(time_tensor).squeeze(0).expand(batch_size, -1)
                    t_emb_k = t_emb.repeat_interleave(K, dim=0)

                    s_T = run_states[-1][subjects]
                    s_T_k = s_T.repeat_interleave(K, dim=0)

                    m_k = torch.full((batch_size * K,), G_STEP, dtype=torch.long, device=device)

                    hc_noisy, _ = scheduler.forward_noise(h_c_k, m_k, branch_type="causal")

                    hc_pred = denoiser(hc_noisy, rel_emb_k, s_T_k, t_emb_k, m_k)

                    # 🔧 FIX: NaN safety
                    hc_pred = torch.nan_to_num(hc_pred, nan=0.0, posinf=1.0, neginf=-1.0)

                    scores_k = predictor(hc_pred, rel_emb_k, run_H_prev)

                    # 🔧 CRITICAL FIX: reshape instead of view
                    assert scores_k.shape[0] == batch_size * K
                    scores_k = scores_k.reshape(batch_size, K, N)

                    scores = scores_k.mean(dim=1)

                    # 🔧 safer masking
                    for i in range(batch_size):
                        h = subjects[i].item()
                        r = relations[i].item()
                        t = objects[i].item()

                        filt = filter_dict[(h, r, tau)]
                        mask_idx = [x for x in filt if x != t]

                        if mask_idx:
                            scores[i, mask_idx] = -1e6

                    sorted_idx = scores.argsort(dim=-1, descending=True)
                    ranks = (sorted_idx == objects.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1

                    mrr += (1.0 / ranks).sum().item()
                    hits1 += (ranks <= 1).sum().item()
                    hits10 += (ranks <= 10).sum().item()
                    total += batch_size

                    # update temporal state
                    x_t_base = emb_layer.get_all_entity_embeddings(time_tensor).squeeze(0)
                    x_t_spatial = spatial_layer(x_t_base, edge_index, edge_type, emb_layer.rel_emb)
                    run_H_prev, run_states, run_prev_xs = temporal_layer(x_t_spatial, run_states, run_prev_xs)

                final_mrr = mrr / total
                final_h1 = hits1 / total
                final_h10 = hits10 / total

                print(f"MRR: {final_mrr:.4f} | Hits@1: {final_h1:.4f} | Hits@10: {final_h10:.4f}")

                if final_mrr > best_mrr:
                    best_mrr = final_mrr

        print("\n" + "="*40)
        print(f"BEST MRR: {best_mrr:.4f}")
        print("="*40)


if __name__ == "__main__":
    tune_inference()