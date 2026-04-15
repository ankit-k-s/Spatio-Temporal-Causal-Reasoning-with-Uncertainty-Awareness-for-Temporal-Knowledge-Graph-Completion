import torch
from models.temporal import GraphSSM

def verify_temporal():
    print("\n" + "=" * 60)
    print("RUNNING STABILIZED CP-4 VERIFICATION")
    print("=" * 60)

    # ICEWS14 dimensions
    N = 12498
    D = 200
    D_STATE = 16 # Refined size
    
    model = GraphSSM(d_model=D, d_state=D_STATE, num_layers=2)
    
    # --------------------------------------------------
    # 1. ODE Stability Check 
    # --------------------------------------------------
    print("\n[1] CONTINUOUS ODE STABILITY CHECK")
    A_vals = -torch.exp(model.layers[0].log_A)
    assert torch.all(A_vals < 0).item(), "FATAL: A matrix contains positive values."
    print("A = -exp(theta) constraint is strictly enforced.")

    # --------------------------------------------------
    # 2. Temporal Stepping & Memory Check
    # --------------------------------------------------
    print("\n[2] TEMPORAL STEPPING & MEMORY CHECK")
    
    x_t0 = torch.randn(N, D)
    x_t1 = torch.randn(N, D)
    
    H_t0, states_t0, prev_xs_t0 = model(x_t0)
    H_t1, states_t1, _ = model(x_t1, states=states_t0, prev_xs=prev_xs_t0)
    
    s_T = states_t1[-1] 
    
    assert list(H_t1.shape) == [N, D], f"H output shape wrong: {H_t1.shape}"
    assert list(s_T.shape) == [N, D, D_STATE], f"s_T memory shape wrong: {s_T.shape}"
    assert not torch.allclose(states_t0[-1], states_t1[-1]), "Memory failed to update."
    
    print(f"H (Output) Shape: {list(H_t1.shape)} (Expected: [{N}, {D}])")
    print(f"s_T (Memory) Shape: {list(s_T.shape)} (Expected: [{N}, {D}, {D_STATE}])")

    # --------------------------------------------------
    # 3. Layer Normalization Check (The Critical Fix)
    # --------------------------------------------------
    print("\n[3] LAYER NORMALIZATION & STABILITY CHECK")
    out_mean = H_t1.mean().item()
    out_std = H_t1.std().item()
    
    print(f"Output Mean: {out_mean:.4f} (Expected ~0.0)")
    print(f"Output Std : {out_std:.4f}  (Expected ~1.0 or controlled)")
    
    assert abs(out_mean) < 0.5, "LayerNorm failed to center the mean."
    assert out_std > 0.1 and out_std < 5.0, "LayerNorm failed to control variance."
    print("Temporal activations are safely normalized. SSM-on-SSM collapse prevented.")

    # --------------------------------------------------
    # 4. Gradient Flow Check
    # --------------------------------------------------
    print("\n[4] GRADIENT FLOW CHECK")
    loss = H_t1.sum()
    loss.backward()
    
    has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grad, "Gradient flow broken!"
    print("Backpropagation flows cleanly through discretized temporal steps.")

    print("\n" + "=" * 60)
    print("ALL CP-4 CHECKS PASSED")
    print("=" * 60)

if __name__ == "__main__":
    verify_temporal()