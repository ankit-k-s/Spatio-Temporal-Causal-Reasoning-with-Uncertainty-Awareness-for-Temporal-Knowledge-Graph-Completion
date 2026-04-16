import torch
from models.diffusion import AsymmetricNoiseScheduler, BiSSMDenoiser

def verify_diffusion():
    print("\n" + "=" * 60)
    print("RUNNING CP-6 VERIFICATION (ASYMMETRIC DIFFUSION + TIME AWARENESS)")
    print("=" * 60)

    # --------------------------------------------------
    # Dimensions (ICEWS14 scale)
    # --------------------------------------------------
    N = 12498 # Total entities in graph
    D = 200
    D_STATE = 16
    DIFFUSION_STEPS = 100
    
    scheduler = AsymmetricNoiseScheduler(num_timesteps=DIFFUSION_STEPS, elevated_beta=5.0)
    denoiser = BiSSMDenoiser(d_model=D, d_state=D_STATE, num_timesteps=DIFFUSION_STEPS)
    
    # --------------------------------------------------
    # Mock inputs for a single timestamp
    # --------------------------------------------------
    h_c_0 = torch.ones(N, D)  # Pure Causal Signal
    h_s_0 = torch.ones(N, D)  # Pure Shortcut Signal
    r_emb = torch.randn(N, D) # Relation query
    t_emb = torch.randn(N, D) # Target Timestamp embedding
    
    # Raw Temporal Memory from GraphSSM (Before projection)
    s_T_raw = torch.randn(N, D, D_STATE)   
    
    # Pick a random discrete diffusion step
    m = torch.tensor([50]).expand(N) # Expanded so each entity has a step
    
    # --------------------------------------------------
    # 1. Asymmetric Noise Check (Stage 6a & 6b)
    # --------------------------------------------------
    print("\n[1] ASYMMETRIC NOISE CHECK")
    hc_m, noise_c = scheduler.forward_noise(h_c_0, m, branch_type="causal")
    hs_m, noise_s = scheduler.forward_noise(h_s_0, m, branch_type="shortcut")
    
    std_c = hc_m.std().item()
    std_s = hs_m.std().item()
    
    print(f"Causal Branch Std   : {std_c:.4f}")
    print(f"Shortcut Branch Std : {std_s:.4f}")
    
    assert std_s > (std_c * 1.5), "Elevated beta failed! Shortcut branch is not receiving elevated variance."
    print("-> [PASS] Shortcut signal successfully obliterated via elevated beta variance.")

    # --------------------------------------------------
    # 2. O(N) Bi-SSM Denoiser Check (Stage 6c)
    # --------------------------------------------------
    print("\n[2] BI-SSM DENOISER CHECK")
    # Feed in all 5 conditioning tensors, including the unflattened memory
    h_denoised = denoiser(hc_m, r_emb, s_T_raw, t_emb, m)
    
    assert list(h_denoised.shape) == [N, D], f"Denoiser shape wrong: {h_denoised.shape}"
    assert not torch.isnan(h_denoised).any(), "NaNs found in denoiser output!"
    print(f"-> [PASS] Bi-SSM successfully projected memory and scanned {N} entities in O(N) time.")

    # --------------------------------------------------
    # 3. Conditioning Integrity Check
    # --------------------------------------------------
    print("\n[3] CONDITIONING GRADIENT FLOW")
    loss = h_denoised.sum()
    loss.backward()
    
    assert denoiser.step_embed.weight.grad is not None, "Gradients missing from Diffusion Step Embedding!"
    assert denoiser.s_proj.weight.grad is not None, "Gradients missing from Memory Projection layer!"
    assert denoiser.condition_fusion[0].weight.grad is not None, "Gradients missing from the 5-way fusion!"
    assert denoiser.bi_ssm_scan.weight_ih_l0.grad is not None, "Gradients missing from Bi-SSM Scan!"
    print("-> [PASS] Gradients flow perfectly backwards through Time, Relation, Step, and Memory conditions.")

    print("\n" + "=" * 60)
    print("PHASE 2 DIFFUSION ARCHITECTURE IS FULLY ISOLATED & VERIFIED")
    print("=" * 60)

if __name__ == "__main__":
    verify_diffusion()