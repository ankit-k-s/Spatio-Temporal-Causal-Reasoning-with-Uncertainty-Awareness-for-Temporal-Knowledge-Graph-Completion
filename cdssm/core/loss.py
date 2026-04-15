import torch
import torch.nn as nn
import torch.nn.functional as F

class QuadObjectiveLoss(nn.Module):
    """
    The Phase 1 Training Engine.
    Forces the model to isolate causal signals and destroy shortcut correlations.
    """
    def __init__(self, lambda_do=0.1, lambda_unif=0.1, lambda_mask=0.01):
        super().__init__()
        self.lambda_do = lambda_do
        self.lambda_unif = lambda_unif
        self.lambda_mask = lambda_mask
        
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, scores_c, scores_do, scores_s, mask, targets):
        """
        scores_c: Logits from Causal branch (h_c)
        scores_do: Logits from Intervened branch (h_c + permuted h_s)
        scores_s: Logits from Shortcut branch (h_s)
        mask: The (0, 1) causal mask values
        targets: The ground truth entity IDs [Batch]
        """
        # 1. Primary Ranking Loss (Train h_c to predict accurately)
        l_rank = self.ce_loss(scores_c, targets)
        
        # 2. Causal Do-Operator Loss (Train model to ignore permuted spurious features)
        l_do = self.ce_loss(scores_do, targets)
        
        # 3. Uniformity Loss (Force h_s to predict pure noise/uniform distribution)
        # We want the shortcut branch to have ZERO predictive power
        N_entities = scores_s.size(-1)
        log_probs_s = F.log_softmax(scores_s, dim=-1)
        uniform_target = torch.full_like(log_probs_s, 1.0 / N_entities)
        
        # KL Divergence measures how far h_s's predictions are from a perfectly flat line
        l_unif = F.kl_div(log_probs_s, uniform_target, reduction='batchmean')
        
        # 4. Mask Entropy Regularization (Prevent mask from collapsing to 0.5)
        # mask * (1 - mask) is minimized when mask is near 0 or 1
        l_mask = torch.mean(mask * (1.0 - mask))
        
        # Total Weighted Loss
        total_loss = l_rank + (self.lambda_do * l_do) + (self.lambda_unif * l_unif) + (self.lambda_mask * l_mask)
        
        metrics = {
            'loss_total': total_loss.item(),
            'loss_rank': l_rank.item(),
            'loss_do': l_do.item(),
            'loss_unif': l_unif.item(),
            'loss_mask': l_mask.item()
        }
        
        return total_loss, metrics