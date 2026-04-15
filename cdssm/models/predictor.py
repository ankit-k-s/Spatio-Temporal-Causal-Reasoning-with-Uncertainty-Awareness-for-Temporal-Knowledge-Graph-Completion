import torch
import torch.nn as nn

class TKGCFusingPredictor(nn.Module):
    """
    Phase 1 Scoring Head.
    Combines Subject History and Relation Query to rank all possible Object Entities.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Relation projection
        self.W_r = nn.Linear(d_model, d_model, bias=False)
        
        # Fusing MLP to combine Subject + Relation
        self.fuse = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, h_subj: torch.Tensor, r_emb: torch.Tensor, all_ent_emb: torch.Tensor):
        q = self.fuse(h_subj + self.W_r(r_emb)) # [Batch, D]
        
        # MEMORY FIX: If all_ent_emb is [N, D] (shared candidate pool), use standard matmul
        if all_ent_emb.dim() == 2:
            scores = torch.matmul(q, all_ent_emb.T) # [Batch, N]
        else:
            # Fallback for batched candidates
            scores = torch.bmm(q.unsqueeze(1), all_ent_emb.transpose(1, 2)).squeeze(1)
            
        return scores