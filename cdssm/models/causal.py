import torch
import torch.nn as nn

class CausalDecomposer(nn.Module):
    """
    Stage 5: Query-Guided CSI Post-Encoding Decomposition.
    Uses Low-Rank Modulation to achieve query-dependence without O(N*R) explosion.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # 1. Base Mask (Entity History)
        self.base_mask_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2), 
            nn.SiLU(), 
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid() 
        )
        
        # 2. Relation Gate (Query Spotlight)
        self.rel_gate_proj = nn.Linear(d_model, d_model)

    def forward(self, h: torch.Tensor, r_query: torch.Tensor):
        """
        h: Temporal node states [N, D]
        r_query: Relation embedding for the current query [Batch, D] or [D]
        """
        # 1. Compute Base Mask (Global Causal Structure)
        # Shape: [N, D]
        base_mask = self.base_mask_mlp(h)
        
        # 2. Compute Relation Gate (Query-Specific Spotlight)
        # Shape: [Batch, D] or [1, D]
        # We use Sigmoid to keep the modulation bounded between (0, 1)
        rel_gate = torch.sigmoid(self.rel_gate_proj(r_query))
        
        # 3. Low-Rank Modulation (The Upgrade)
        # We unsqueeze rel_gate to broadcast across all N entities safely if batched
        if rel_gate.dim() == 1:
            rel_gate = rel_gate.unsqueeze(0) # [1, D]
            
        # Shape: [Batch, N, D] or [N, D] depending on inputs
        # If processing a single relation against N entities, it broadcasts naturally
        mask = base_mask * rel_gate
        
        # 4. Hadamard Split
        h_c = mask * h
        h_s = (1.0 - mask) * h
        
        return h_c, h_s, mask

    def intervene(self, h_s: torch.Tensor) -> torch.Tensor:
        """
        The do-operator (do(C)). Shuffles the shortcut representations.
        """
        perm = torch.randperm(h_s.size(0), device=h_s.device)
        return h_s[perm]