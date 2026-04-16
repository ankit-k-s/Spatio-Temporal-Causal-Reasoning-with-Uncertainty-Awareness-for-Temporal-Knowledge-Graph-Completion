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
    filter_dict = defaultdict(set)
    for split in [loader.train_snapshots, loader.valid_snapshots, loader.test_snapshots]:
        for tau, (edge_index, edge_type, _) in split.items():
            heads, tails, rels = edge_index[0].tolist(), edge_index[1].tolist(), edge_type.tolist()
            for h, r, t in zip(heads, rels, tails):
                filter_dict[(h, r, tau)].add(t)
    return filter_dict

def run_strict_evaluation():
    print("\n" + "="*60)
    print("CD-SSM LEAKAGE & ABLATION VERIFICATION (TEST SET)")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = TKGDataloader(data_dir="data/ICEWS14")
    filter_dict = build_filter_dict(loader)
    
    N, D, D_STATE = loader.num_entities, 200, 16

    emb_layer = TKGEmbedding(N, loader.num_relations_total, len(loader.time2id), D).to(device)
    spatial_layer = GraphMambaLayer(d_model=D, seq_len=3).to(device)
    temporal_layer = GraphSSM(d_model=D, d_state=D_STATE, num_layers=2).to(device)
    decomposer = CausalDecomposer(d_model=D).to(device)
    predictor = TKGCFusingPredictor(d_model=D).to(device)

    # Load Weights
    checkpoint = torch.load("checkpoints/cdssm_phase1.pt", map_location=device, weights_only=True)
    emb_layer.load_state_dict(checkpoint['emb_layer'])
    spatial_layer.load_state_dict(checkpoint['spatial_layer'])
    temporal_layer.load_state_dict(checkpoint['temporal_layer'])
    decomposer.load_state_dict(checkpoint['decomposer'])
    predictor.load_state_dict(checkpoint['predictor'])

    for m in [emb_layer, spatial_layer, temporal_layer, decomposer, predictor]: m.eval()

    with torch.no_grad():
        print("\n[1] Rolling Temporal Memory (Train + Valid)...")
        states, prev_xs = None, None
        
        # Warmup through Train AND Valid to prep for Test set
        warmup_times = sorted(list(loader.train_snapshots.keys()) + list(loader.valid_snapshots.keys()))
        all_snapshots = {**loader.train_snapshots, **loader.valid_snapshots, **loader.test_snapshots}
        
        # We need to track the LAST known H_t to use for predicting the future
        H_prev = torch.zeros(N, D).to(device)
        
        for tau in warmup_times:
            edge_index, edge_type, time_id = all_snapshots[tau]
            edge_index, edge_type = edge_index.to(device), edge_type.to(device)
            
            x_t_base = emb_layer.get_all_entity_embeddings(time_id).squeeze(0)
            x_t_spatial = spatial_layer(x_t_base, edge_index, edge_type, emb_layer.rel_emb)
            H_prev, states, prev_xs = temporal_layer(x_t_spatial, states, prev_xs)

        print("\n[2] STRICT EVALUATION: TEST SET (No Leaks)")
        mrr, hits_1, hits_10, total_queries = 0.0, 0.0, 0.0, 0
        
        test_times = sorted(loader.test_snapshots.keys())
        for tau in tqdm(test_times, desc="Testing"):
            edge_index, edge_type, time_id = loader.test_snapshots[tau]
            edge_index, edge_type = edge_index.to(device), edge_type.to(device)
            
            subjects, objects, relations = edge_index[0], edge_index[1], edge_type
            
            # ---------------------------------------------------------
            # STRICT FORECASTING: Predict using H_prev (from time t-1)
            # We DO NOT process the current edge_index before predicting!
            # ---------------------------------------------------------
            subj_states = H_prev[subjects]
            rel_embeddings = emb_layer.rel_emb(relations)
            
            # Causal split
            h_c, _, mask = decomposer(subj_states, rel_embeddings)
            
            # Check 1: Candidate Size Check
            scores = predictor(h_c, rel_embeddings, H_prev) 
            if tau == test_times[0]:
                print(f"\n -> Candidate Space Shape: {list(scores.shape)} (Expected: [Batch, 12498])")

            # Filtered protocol
            for i in range(len(subjects)):
                h, r, true_tail = subjects[i].item(), relations[i].item(), objects[i].item()
                valid_tails = filter_dict[(h, r, tau)]
                mask_idx = [t for t in valid_tails if t != true_tail]
                if mask_idx:
                    scores[i, mask_idx] = -1e9
            
            sorted_indices = scores.argsort(dim=-1, descending=True)
            ranks = (sorted_indices == objects.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1.0
            
            mrr += (1.0 / ranks).sum().item()
            hits_1 += (ranks <= 1).sum().item()
            hits_10 += (ranks <= 10).sum().item()
            total_queries += len(subjects)

            # ---------------------------------------------------------
            # AFTER predicting, we process time t to update memory for t+1
            # ---------------------------------------------------------
            x_t_base = emb_layer.get_all_entity_embeddings(time_id).squeeze(0)
            x_t_spatial = spatial_layer(x_t_base, edge_index, edge_type, emb_layer.rel_emb)
            H_prev, states, prev_xs = temporal_layer(x_t_spatial, states, prev_xs)

        print("\n" + "="*40)
        print("====== TRUE BASELINE RESULTS ======")
        print("="*40)
        print(f"Strict Filtered MRR : {mrr / total_queries:.4f}")
        print(f"Strict Hits@1       : {hits_1 / total_queries:.4f}")
        print(f"Strict Hits@10      : {hits_10 / total_queries:.4f}")
        print("="*40 + "\n")

if __name__ == "__main__":
    run_strict_evaluation()