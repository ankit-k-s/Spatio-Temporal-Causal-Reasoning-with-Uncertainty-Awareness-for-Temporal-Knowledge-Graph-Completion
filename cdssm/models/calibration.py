import torch
import torch.nn as nn

class ConfidenceHead(nn.Module):
    """
    Learns to dynamically gate the frequency prior based on model uncertainty.
    """
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1),
            nn.Sigmoid() # Outputs a value between 0 (highly confident) and 1 (uncertain)
        )

    def forward(self, hc_pred, rel_emb):
        # Concatenate the denoised state and the relation query
        x = torch.cat([hc_pred, rel_emb], dim=-1)
        return self.mlp(x)