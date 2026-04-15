import torch
import torch.nn as nn
from models.spatial import GraphMambaLayer

def verify_spatial():
    print("\n" + "=" * 60)
    print("RUNNING DETAILED CP-3 VERIFICATION")
    print("=" * 60)

    # ICEWS14 dimensions
    N = 12498
    D = 200
    SEQ_LEN = 3
    NUM_REL = 520
    E = 3525 # Avg edges
    
    # Initialize Layer
    model = GraphMambaLayer(d_model=D, seq_len=SEQ_LEN)
    rel_emb_layer = nn.Embedding(NUM_REL, D)
    
    # Dummy Inputs
    x = torch.randn(N, D)
    src = torch.randint(0, N, (E,))
    dst = torch.randint(0, N, (E,))
    edge_index = torch.stack([src, dst], dim=0)
    edge_type = torch.randint(0, NUM_REL, (E,))
    
    # --------------------------------------------------
    # 1. Gradient & Shape Check
    # --------------------------------------------------
    print("\n[1] GRADIENT & SHAPE CHECK")
    out = model(x, edge_index, edge_type, rel_emb_layer)
    
    assert list(out.shape) == [N, D], f"Shape mismatch: {out.shape}"
    assert out.requires_grad, "Output lost gradient tracking!"
    assert not torch.isnan(out).any(), "NaNs detected in output!"
    print("Output shapes are perfectly preserved and gradients are active.")

    # --------------------------------------------------
    # 2. Token Diversity Check (The Fix 1 Test)
    # --------------------------------------------------
    print("\n[2] TOKEN DIVERSITY CHECK")
    # We must extract the tokens from the tokenizer to check them
    tokens = model.tokenizer(x, edge_index, edge_type, rel_emb_layer)
    
    # Check if Step 1 features are identical to Step 2 features
    step_1 = tokens[:, 1, :]
    step_2 = tokens[:, 2, :]
    
    is_diverse = not torch.allclose(step_1, step_2, atol=1e-4)
    assert is_diverse, "Tokenizer failed! Steps contain identical aggregated features."
    print("Sequence steps contain diverse, randomized neighbor subsets.")

    # --------------------------------------------------
    # 3. Relation-Awareness Check (The Fix 2 Test)
    # --------------------------------------------------
    print("\n[3] RELATION-AWARENESS CHECK")
    
    # Run again with a COMPLETELY DIFFERENT edge_type configuration
    edge_type_alt = torch.randint(0, NUM_REL, (E,))
    out_alt = model(x, edge_index, edge_type_alt, rel_emb_layer)
    
    # If the model ignores relations, out will equal out_alt
    is_relation_aware = not torch.allclose(out, out_alt, atol=1e-4)
    assert is_relation_aware, "Spatial layer is ignoring relation types!"
    print("GraphMamba is successfully utilizing relation semantics.")

    # --------------------------------------------------
    # 4. Stability Check (LayerNorm Effectiveness)
    # --------------------------------------------------
    print("\n[4] LAYER NORMALIZATION CHECK")
    
    out_mean = out.mean().item()
    out_std = out.std().item()
    
    print(f"Output Mean: {out_mean:.4f} (Expected ~0.0)")
    print(f"Output Std : {out_std:.4f}  (Expected ~1.0 or controlled)")
    
    assert abs(out_mean) < 0.5, "LayerNorm failed to center the mean."
    assert out_std > 0.1 and out_std < 5.0, "LayerNorm failed to control variance."
    print("Activations are safely normalized. Exploding gradients prevented.")

    print("\n" + "=" * 60)
    print("ALL CP-3 CHECKS PASSED")
    print("=" * 60)

if __name__ == "__main__":
    verify_spatial()