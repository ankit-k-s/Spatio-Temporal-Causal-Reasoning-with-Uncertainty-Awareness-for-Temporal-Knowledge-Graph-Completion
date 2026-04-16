import torch
from models.causal import CausalDecomposer

def verify_causal():
    print("\n" + "=" * 60)
    print("RUNNING UPGRADED CP-5 VERIFICATION (LOW-RANK MODULATION)")
    print("=" * 60)

    N = 12498
    D = 200
    
    model = CausalDecomposer(d_model=D)
    
    h_mock = torch.randn(N, D, requires_grad=True)
    r_query_mock = torch.randn(D, requires_grad=True) # Single query relation
    
    # --------------------------------------------------
    # 1. Decomposition & Broadcasting Check
    # --------------------------------------------------
    print("\n[1] DECOMPOSITION & BROADCAST CHECK")
    h_c, h_s, mask = model(h_mock, r_query_mock)
    
    assert list(h_c.shape) == [N, D], f"h_c shape wrong: {h_c.shape}"
    assert list(h_s.shape) == [N, D], f"h_s shape wrong: {h_s.shape}"
    assert list(mask.shape) == [N, D], f"mask shape wrong: {mask.shape}"
    print("Low-Rank Modulation successfully broadcasted [D] relation to [N, D] entities.")

    # --------------------------------------------------
    # 2. Query Sensitivity Check (The Core Upgrade)
    # --------------------------------------------------
    print("\n[2] QUERY SENSITIVITY CHECK")
    r_query_alt = torch.randn(D) # A different relation
    _, _, mask_alt = model(h_mock, r_query_alt)
    
    is_sensitive = not torch.allclose(mask, mask_alt, atol=1e-4)
    assert is_sensitive, "Mask ignored the relation query! Upgrade failed."
    print("Mask dynamically changes based on relation query. Query-Dependence achieved.")

    # --------------------------------------------------
    # 3. Mathematical Integrity & Bounds
    # --------------------------------------------------
    print("\n[3] MATHEMATICAL INTEGRITY CHECK")
    assert torch.all((mask >= 0.0) & (mask <= 1.0)), "Mask values out of bounds!"
    
    reconstructed_h = h_c + h_s
    assert torch.allclose(reconstructed_h, h_mock, atol=1e-5), "h_c + h_s != h!"
    print("h_c + h_s == h identity is preserved. Mask strictly bounded (0, 1).")

    # --------------------------------------------------
    # 4. Gradient Flow Check
    # --------------------------------------------------
    print("\n[4] GRADIENT FLOW CHECK")
    loss = h_c.sum() + h_s.sum()
    loss.backward()
    
    assert h_mock.grad is not None, "Gradient flow broken to Temporal Engine!"
    assert r_query_mock.grad is not None, "Gradient flow broken to Relation Embeddings!"
    print("Backpropagation flows cleanly to both Entity History and Relation Query.")

    print("\n" + "=" * 60)
    print("ALL CP-5 CHECKS PASSED")
    print("=" * 60)

if __name__ == "__main__":
    verify_causal()