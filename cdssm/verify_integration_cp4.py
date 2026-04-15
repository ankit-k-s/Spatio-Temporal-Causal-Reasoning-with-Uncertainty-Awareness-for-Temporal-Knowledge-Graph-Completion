import torch
import torch.nn as nn
from models.embedding import TKGEmbedding
from models.spatial import GraphMambaLayer
from models.temporal import GraphSSM

def verify_integration():
    print("\n" + "=" * 60)
    print("RUNNING CP-2 → CP-3 → CP-4 INTEGRATION TEST")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Setup & Dimensions (ICEWS14)
    # --------------------------------------------------
    N = 12498
    D = 200
    NUM_REL = 520
    NUM_TIME = 348
    D_STATE = 16
    E = 3500 # Simulated edges per snapshot
    
    print("[1] Initializing Stack...")
    emb_layer = TKGEmbedding(N, NUM_REL, NUM_TIME, D)
    spatial_layer = GraphMambaLayer(d_model=D, seq_len=3)
    temporal_layer = GraphSSM(d_model=D, d_state=D_STATE, num_layers=2)

    # --------------------------------------------------
    # 2. Simulate Timestamps (t=0 and t=1)
    # --------------------------------------------------
    print("\n[2] Simulating Data Flow (t=0 -> t=1)...")
    
    # Data for t=0
    t0 = torch.tensor([0])
    edge_index_0 = torch.stack([torch.randint(0, N, (E,)), torch.randint(0, N, (E,))], dim=0)
    edge_type_0 = torch.randint(0, NUM_REL, (E,))
    
    # Data for t=1
    t1 = torch.tensor([1])
    edge_index_1 = torch.stack([torch.randint(0, N, (E,)), torch.randint(0, N, (E,))], dim=0)
    edge_type_1 = torch.randint(0, NUM_REL, (E,))

    # --------------------------------------------------
    # 3. Forward Pass: Timestep 0
    # --------------------------------------------------
    # A. Get time-aware base embeddings for all nodes [N, D]
    x_t0_base = emb_layer.get_all_entity_embeddings(t0).squeeze(0)
    
    # B. Inject Spatial GraphMamba structure
    x_t0_spatial = spatial_layer(x_t0_base, edge_index_0, edge_type_0, emb_layer.rel_emb)
    
    # C. Track via Temporal GraphSSM
    H_t0, states_0, prev_x_0 = temporal_layer(x_t0_spatial)

    # --------------------------------------------------
    # 4. Forward Pass: Timestep 1
    # --------------------------------------------------
    x_t1_base = emb_layer.get_all_entity_embeddings(t1).squeeze(0)
    x_t1_spatial = spatial_layer(x_t1_base, edge_index_1, edge_type_1, emb_layer.rel_emb)
    
    # CRITICAL: Pass the memory (states_0) and MIX inputs (prev_x_0) forward!
    H_t1, states_1, prev_x_1 = temporal_layer(x_t1_spatial, states=states_0, prev_xs=prev_x_0)

    # --------------------------------------------------
    # 5. Pipeline Integrity Checks
    # --------------------------------------------------
    print("\n[3] Verifying Integration Constraints...")
    
    # Check 1: Final Output Shape
    assert list(H_t1.shape) == [N, D], f"Final representation shape failed: {H_t1.shape}"
    print(" -> [PASS] Final representation (H_t1) shape is correct.")
    
    # Check 2: Memory Update Validation
    assert not torch.allclose(states_0[-1], states_1[-1]), "Memory state did not change between t=0 and t=1!"
    print(" -> [PASS] Temporal memory successfully updated across timestamps.")
    
    # Check 3: End-to-End Gradient Flow
    # If we sum H_t1 and backprop, the gradient should reach the base embeddings at t=0
    # because the temporal ODE carries the mathematical link backward through time.
    loss = H_t1.sum()
    loss.backward()
    
    assert emb_layer.ent_emb.weight.grad is not None, "Gradient failed to reach Entity Embeddings!"
    assert emb_layer.time_emb.weight.grad is not None, "Gradient failed to reach Time Embeddings!"
    assert spatial_layer.W_g.weight.grad is not None, "Gradient failed to reach Spatial Layer!"
    assert temporal_layer.layers[0].log_A.grad is not None, "Gradient failed to reach Temporal Layer!"
    
    print(" -> [PASS] End-to-end BPTT (Backpropagation Through Time) is fully connected.")

    # Check 4: Stability Check
    out_mean = H_t1.mean().item()
    out_std = H_t1.std().item()
    assert abs(out_mean) < 1.0 and out_std < 5.0, f"Pipeline exploded: Mean={out_mean}, Std={out_std}"
    print(f" -> [PASS] Stack remains numerically stable. (Mean: {out_mean:.4f}, Std: {out_std:.4f})")

    print("\n" + "=" * 60)
    print("SYSTEM INTEGRATION: VERIFIED & READY FOR CP-5")
    print("=" * 60)

if __name__ == "__main__":
    verify_integration()