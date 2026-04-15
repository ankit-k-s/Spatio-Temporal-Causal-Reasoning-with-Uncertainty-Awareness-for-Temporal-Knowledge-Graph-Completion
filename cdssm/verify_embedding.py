import torch
from models.embedding import TKGEmbedding

def verify_embedding():
    print("\n" + "=" * 60)
    print("RUNNING DETAILED CP-2 VERIFICATION")
    print("=" * 60)

    # Use actual ICEWS14 stats from CP-1
    NUM_ENT = 12498
    NUM_REL = 520
    NUM_TIME = 348
    DIM = 200
    BATCH_SIZE = 8

    # Initialize model
    model = TKGEmbedding(NUM_ENT, NUM_REL, NUM_TIME, DIM)

    # --------------------------------------------------
    # 1. Gradient & Initialization Check
    # --------------------------------------------------
    print("\n[1] GRADIENT & INITIALIZATION CHECK")
    
    weights = [
        ("Entity", model.ent_emb.weight),
        ("Relation", model.rel_emb.weight),
        ("Time", model.time_emb.weight),
        ("Time Proj", model.time_proj.weight)
    ]
    
    for name, w in weights:
        assert w.requires_grad, f"{name} embedding does not require gradients!"
        assert not torch.all(w == 0), f"{name} embedding initialized to all zeros!"
        
    print("All parameters correctly initialized and tracking gradients.")

    # --------------------------------------------------
    # 2. Time-Awareness Check (The Core Design)
    # --------------------------------------------------
    print("\n[2] TIME-AWARENESS CHECK")
    
    # Same entity, same relation, different times
    s = torch.tensor([5, 5])
    r = torch.tensor([10, 10])
    o = torch.tensor([20, 20])
    t = torch.tensor([0, 100]) # Time 0 vs Time 100

    e_s, e_r, e_o, _ = model(s, r, o, t)

    # Entity embeddings MUST change across time
    assert not torch.allclose(e_s[0], e_s[1]), "Subject embedding failed to modulate with time!"
    assert not torch.allclose(e_o[0], e_o[1]), "Object embedding failed to modulate with time!"
    
    # Relation embeddings MUST remain static across time
    assert torch.allclose(e_r[0], e_r[1]), "Relation embedding incorrectly modulated by time!"

    print("Entity embeddings successfully dynamically modulate with time.")
    print("Relation embeddings successfully remain time-invariant.")

    # --------------------------------------------------
    # 3. Mathematical Equivalence Check
    # --------------------------------------------------
    print("\n[3] MATHEMATICAL EQUIVALENCE CHECK")
    
    test_s = torch.tensor([42])
    test_t = torch.tensor([10])
    
    # Forward pass
    e_s_forward, _, _, _ = model(test_s, test_s, test_s, test_t)
    
    # Manual calculation: E_ent[s] + W_t * E_time[t]
    manual_base = model.ent_emb(test_s)
    manual_time = model.time_proj(model.time_emb(test_t))
    manual_combined = manual_base + manual_time
    
    assert torch.allclose(e_s_forward, manual_combined, atol=1e-6), "Forward pass math does not match formulation!"
    
    print("Forward pass perfectly matches theoretical formulation: e_s(t) = E_ent[s] + W_t * E_time[t]")

    # --------------------------------------------------
    # 4. Broadcast Equivalence Check (For Eval)
    # --------------------------------------------------
    print("\n[4] EVALUATION BROADCAST CHECK")
    
    # Get all entities evaluated at time 10
    all_ents_at_t = model.get_all_entity_embeddings(torch.tensor([10])) # Shape: [1, N, D]
    
    # Compare with entity 42 evaluated explicitly at time 10
    extracted_entity_42 = all_ents_at_t[0, 42, :]
    
    assert torch.allclose(e_s_forward.squeeze(), extracted_entity_42, atol=1e-6), "Broadcast embedding does not match batched embedding!"
    
    print("O(1) Broadcast matrix perfectly matches individual O(N) forward passes.")

    print("\n" + "=" * 60)
    print("ALL EMBEDDING CHECKS PASSED")
    print("=" * 60)

if __name__ == "__main__":
    verify_embedding()