import torch
from collections import defaultdict
from tqdm import tqdm

from data.loader import TKGDataloader
from models.embedding import TKGEmbedding
from models.spatial import GraphMambaLayer
from models.temporal import GraphSSM
from models.causal import CausalDecomposer
from models.predictor import TKGCFusingPredictor

def build_filter_dict(loader):
    """Builds a dictionary of all true (head, rel, tau) -> set(tails) to prevent penalizing valid alternative facts."""
    filter_dict = defaultdict(set)
    for split in [loader.train_snapshots, loader.valid_snapshots, loader.test_snapshots]:
        for tau, (edge_index, edge_type, _) in split.items():
            heads, tails = edge_index[0].tolist(), edge_index[1].tolist()
            rels = edge_type.tolist()
            for h, r, t in zip(heads, rels, tails):
                filter_dict[(h, r, tau)].add(t)
    return filter_dict

def evaluate_phase1():
    print("\n" + "="*60)
    print("CD-SSM PHASE 1 SOTA EVALUATION (STRICT FORECASTING)")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data & Filter
    loader = TKGDataloader(data_dir="data/ICEWS14")
    filter_dict = build_filter_dict(loader)
    
    # --- CHAMPION HYPERPARAMS (RESTORED) ---
    N = loader.num_entities
    D = 200        
    D_STATE = 16   
    print(f"Evaluating on device: {device} | D={D} | D_STATE={D_STATE}")

    # 2. Instantiate Architecture
    emb_layer = TKGEmbedding(N, loader.num_relations_total, len(loader.time2id), D).to(device)
    spatial_layer = GraphMambaLayer(d_model=D, seq_len=3).to(device)
    temporal_layer = GraphSSM(d_model=D, d_state=D_STATE, num_layers=2).to(device)
    decomposer = CausalDecomposer(d_model=D).to(device)
    predictor = TKGCFusingPredictor(d_model=D).to(device)

    # 3. Load Saved Weights
    try:
        checkpoint = torch.load("checkpoints/cdssm_phase1.pt", map_location=device, weights_only=True)
        emb_layer.load_state_dict(checkpoint['emb_layer'])
        spatial_layer.load_state_dict(checkpoint['spatial_layer'])
        temporal_layer.load_state_dict(checkpoint['temporal_layer'])
        decomposer.load_state_dict(checkpoint['decomposer'])
        predictor.load_state_dict(checkpoint['predictor'])
        print("✅ D=200 Checkpoint loaded successfully.")
    except FileNotFoundError:
        print("❌ Error: 'checkpoints/cdssm_phase1.pt' not found!")
        return

    # Set to eval mode
    for m in [emb_layer, spatial_layer, temporal_layer, decomposer, predictor]: m.eval()

    with torch.no_grad():
        # 4. WARMUP: Roll temporal memory through Train + Valid
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

        # 5. EVALUATION: Calculate metrics on Test Set
        print("\n[2] Evaluating Test Set (Strict Forecasting)...")
        mrr, hits_1, hits_3, hits_10, total_queries = 0.0, 0.0, 0.0, 0.0, 0
        
        test_times = sorted(loader.test_snapshots.keys())
        for tau in tqdm(test_times, desc="Testing"):
            edge_index, edge_type, time_id = loader.test_snapshots[tau]
            edge_index, edge_type = edge_index.to(device), edge_type.to(device)
            time_tensor = torch.tensor([time_id], dtype=torch.long, device=device)
            
            subjects, objects, relations = edge_index[0], edge_index[1], edge_type
            
            # --- 1. PREDICT USING YESTERDAY'S MEMORY (H_prev) ---
            subj_states = H_prev[subjects]
            rel_embeddings = emb_layer.rel_emb(relations)
            
            # Use the exact Do-Intervention logic from your Hybrid script
            h_c, h_s, _ = decomposer(subj_states, rel_embeddings)
            h_do = h_c + decomposer.intervene(h_s)
            scores = predictor(h_do, rel_embeddings, H_prev) 
            
            # --- 2. FILTER & SCORE ---
            for i in range(len(subjects)):
                h, r, true_tail = subjects[i].item(), relations[i].item(), objects[i].item()
                valid_tails = filter_dict[(h, r, tau)]
                mask_idx = [t for t in valid_tails if t != true_tail]
                if mask_idx:
                    scores[i, mask_idx] = -1e6 # Safe masking
            
            sorted_indices = scores.argsort(dim=-1, descending=True)
            ranks = (sorted_indices == objects.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1.0
            
            mrr += (1.0 / ranks).sum().item()
            hits_1 += (ranks <= 1).sum().item()
            hits_3 += (ranks <= 3).sum().item()
            hits_10 += (ranks <= 10).sum().item()
            total_queries += len(subjects)

            # --- 3. UPDATE MEMORY WITH TODAY'S FACTS ---
            x_t_base = emb_layer.get_all_entity_embeddings(time_tensor).squeeze(0)
            x_t_spatial = spatial_layer(x_t_base, edge_index, edge_type, emb_layer.rel_emb)
            H_prev, states, prev_xs = temporal_layer(x_t_spatial, states, prev_xs)

        print("\n" + "="*40)
        print("====== PHASE 1 RESULTS (D=200 + Margin Loss) ======")
        print(f"Target Baseline : ~0.2227")
        print("-" * 40)
        print(f"Filtered MRR    : {mrr / total_queries:.4f}")
        print(f"Hits@1          : {hits_1 / total_queries:.4f}")
        print(f"Hits@3          : {hits_3 / total_queries:.4f}")
        print(f"Hits@10         : {hits_10 / total_queries:.4f}")
        print("="*40 + "\n")

if __name__ == "__main__":
    evaluate_phase1()