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

def evaluate_model():
    print("\n" + "="*50)
    print("CD-SSM PHASE 1 EVALUATION")
    print("="*50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # 1. Load Data & Filter
    loader = TKGDataloader(data_dir="data/ICEWS14")
    filter_dict = build_filter_dict(loader)
    
    N = loader.num_entities
    D, D_STATE = 200, 16

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
        print("✅ Checkpoint loaded successfully.")
    except FileNotFoundError:
        print("❌ Error: 'checkpoints/cdssm_phase1.pt' not found. Run train.py first!")
        return

    # Set to eval mode
    modules = [emb_layer, spatial_layer, temporal_layer, decomposer, predictor]
    for m in modules: m.eval()

    with torch.no_grad():
        # 4. WARMUP: Roll temporal memory through the training set
        print("\n[Phase 1] Rolling Temporal Memory (Warmup)...")
        states, prev_xs = None, None
        
        for tau in sorted(loader.train_snapshots.keys()):
            edge_index, edge_type, time_id = loader.train_snapshots[tau]
            edge_index, edge_type = edge_index.to(device), edge_type.to(device)
            
            x_t_base = emb_layer.get_all_entity_embeddings(time_id).squeeze(0)
            x_t_spatial = spatial_layer(x_t_base, edge_index, edge_type, emb_layer.rel_emb)
            _, states, prev_xs = temporal_layer(x_t_spatial, states, prev_xs)

        # 5. EVALUATION: Calculate metrics on Validation Set
        print("\n[Phase 2] Evaluating Validation Set...")
        mrr, hits_1, hits_3, hits_10, total_queries = 0.0, 0.0, 0.0, 0.0, 0
        
        valid_times = sorted(loader.valid_snapshots.keys())
        for tau in tqdm(valid_times, desc="Validation"):
            edge_index, edge_type, time_id = loader.valid_snapshots[tau]
            edge_index, edge_type = edge_index.to(device), edge_type.to(device)
            
            # Forward pass
            x_t_base = emb_layer.get_all_entity_embeddings(time_id).squeeze(0)
            x_t_spatial = spatial_layer(x_t_base, edge_index, edge_type, emb_layer.rel_emb)
            H_t, states, prev_xs = temporal_layer(x_t_spatial, states, prev_xs)
            
            # Predict
            subjects, objects, relations = edge_index[0], edge_index[1], edge_type
            subj_states = H_t[subjects]
            rel_embeddings = emb_layer.rel_emb(relations)
            
            # We ONLY evaluate the causal branch (h_c) during inference
            h_c, _, _ = decomposer(subj_states, rel_embeddings)
            scores = predictor(h_c, rel_embeddings, H_t) # [Batch, N]
            
            # Filtered Evaluation Protocol
            for i in range(len(subjects)):
                h, r, true_tail = subjects[i].item(), relations[i].item(), objects[i].item()
                
                # Get all valid answers for this query
                valid_tails = filter_dict[(h, r, tau)]
                
                # Mask all valid tails EXCEPT the current true target
                mask_idx = [t for t in valid_tails if t != true_tail]
                if mask_idx:
                    scores[i, mask_idx] = -1e9
            
            # Calculate Ranks
            # argsort(descending=True) gives indices of highest scores first
            sorted_indices = scores.argsort(dim=-1, descending=True)
            ranks = (sorted_indices == objects.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1.0
            
            # Accumulate Metrics
            mrr += (1.0 / ranks).sum().item()
            hits_1 += (ranks <= 1).sum().item()
            hits_3 += (ranks <= 3).sum().item()
            hits_10 += (ranks <= 10).sum().item()
            total_queries += len(subjects)

        # Print Final Results
        print("\n" + "="*40)
        print("====== EVALUATION RESULTS ======")
        print("="*40)
        print(f"Filtered MRR : {mrr / total_queries:.4f}")
        print(f"Hits@1       : {hits_1 / total_queries:.4f}")
        print(f"Hits@3       : {hits_3 / total_queries:.4f}")
        print(f"Hits@10      : {hits_10 / total_queries:.4f}")
        print("="*40 + "\n")

if __name__ == "__main__":
    evaluate_model()