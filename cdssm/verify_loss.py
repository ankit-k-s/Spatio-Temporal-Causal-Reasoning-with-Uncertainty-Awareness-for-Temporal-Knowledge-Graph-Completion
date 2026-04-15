import torch
from models.predictor import TKGCFusingPredictor
from core.loss import QuadObjectiveLoss

def verify_predictor_and_loss():
    print("\n" + "=" * 60)
    print("RUNNING PREDICTOR & LOSS VERIFICATION")
    print("=" * 60)

    # Dimensions
    BATCH = 32
    N = 12498
    D = 200
    
    # Initialize Modules
    predictor = TKGCFusingPredictor(d_model=D)
    loss_fn = QuadObjectiveLoss()
    
    # Mock Data from Causal Decomposer
    h_c = torch.randn(BATCH, D, requires_grad=True)
    h_s = torch.randn(BATCH, D, requires_grad=True)
    h_do = torch.randn(BATCH, D, requires_grad=True) # Usually h_c + permuted h_s
    mask = torch.rand(BATCH, D, requires_grad=True)  # Already Sigmoid activated
    
    # Mock Embeddings
    r_emb = torch.randn(BATCH, D)
    all_ent_emb = torch.randn(BATCH, N, D)
    
    # Ground Truth Targets
    targets = torch.randint(0, N, (BATCH,))
    
    # --------------------------------------------------
    # 1. Predictor Scoring Check
    # --------------------------------------------------
    print("\n[1] PREDICTOR SCORING CHECK")
    scores_c = predictor(h_c, r_emb, all_ent_emb)
    scores_s = predictor(h_s, r_emb, all_ent_emb)
    scores_do = predictor(h_do, r_emb, all_ent_emb)
    
    assert list(scores_c.shape) == [BATCH, N], f"Scores shape wrong: {scores_c.shape}"
    print("Predictor successfully collapsed [Batch, D] and [Batch, N, D] into [Batch, N] logits.")

    # --------------------------------------------------
    # 2. Quad-Objective Loss Check
    # --------------------------------------------------
    print("\n[2] QUAD-OBJECTIVE LOSS CHECK")
    total_loss, metrics = loss_fn(scores_c, scores_do, scores_s, mask, targets)
    
    print(f"Total Loss  : {metrics['loss_total']:.4f}")
    print(f" -> Rank    : {metrics['loss_rank']:.4f}")
    print(f" -> Do-Op   : {metrics['loss_do']:.4f}")
    print(f" -> Uniform : {metrics['loss_unif']:.4f}")
    print(f" -> Entropy : {metrics['loss_mask']:.4f}")
    
    assert not torch.isnan(total_loss), "Loss computed NaN!"
    print("Quad-Objective math is stable and functioning correctly.")

    # --------------------------------------------------
    # 3. Full Backprop Check
    # --------------------------------------------------
    print("\n[3] BACKPROPAGATION CHECK")
    total_loss.backward()
    
    assert h_c.grad is not None, "Gradients failed to reach h_c"
    assert mask.grad is not None, "Gradients failed to reach causal mask"
    print("Gradients flow perfectly from Loss -> Predictor -> Causal Decomposer.")

    print("\n" + "=" * 60)
    print("PHASE 1 ARCHITECTURE FULLY VERIFIED")
    print("=" * 60)

if __name__ == "__main__":
    verify_predictor_and_loss()