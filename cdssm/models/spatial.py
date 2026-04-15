import torch
import torch.nn as nn
from einops import einsum, rearrange

class RelationAwareTokenizer(nn.Module):
    """
    Refined Tokenizer: Randomly subsamples edges per step to create true diversity.
    Crucially injects relation embeddings into the spatial features.
    """
    def __init__(self, seq_len: int = 5):
        super().__init__()
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor, rel_emb: torch.Tensor):
        N, D = x.shape
        device = x.device
        E = edge_index.size(1)
        
        # Initialize token sequences: [N, S, D]
        tokens = torch.zeros((N, self.seq_len, D), device=device)
        tokens[:, 0, :] = x  # The target node is always token 0
        
        src, dst = edge_index
        
        # Ensure we don't try to sample more edges than exist
        edges_per_step = max(1, E // (self.seq_len - 1)) if E > 0 else 0
        
        for i in range(1, self.seq_len):
            if E == 0:
                break # Graph is empty at this timestamp
                
            # 1. Randomly sample a subset of edges to create diversity
            perm = torch.randperm(E, device=device)[:edges_per_step]
            step_src = src[perm]
            step_dst = dst[perm]
            step_rel = edge_type[perm]
            
            # 2. Relation-aware feature construction
            # Message = Source Entity + Relation
            msg = x[step_src] + rel_emb(step_rel)
            
            # 3. Aggregate into destination nodes
            neighbor_feats = torch.zeros_like(x)
            neighbor_feats.scatter_add_(0, step_dst.unsqueeze(1).expand(-1, D), msg)
            
            # 4. Degree Normalization
            degree = torch.zeros(N, device=device).scatter_add_(0, step_dst, torch.ones_like(step_dst, dtype=torch.float))
            degree = degree.clamp(min=1.0).unsqueeze(1)
            
            tokens[:, i, :] = neighbor_feats / degree
            
        return tokens

class PureBiMambaScan(nn.Module):
    """
    Refined BiMamba: Pre-computes discretization outside the loop for massive memory efficiency.
    """
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        self.log_A = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.proj_B = nn.Linear(d_model, d_state, bias=False)
        self.proj_C = nn.Linear(d_model, d_state, bias=False)
        self.proj_Delta = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Softplus()
        )

    def selective_scan(self, tokens: torch.Tensor):
        N, S, D = tokens.shape
        state = torch.zeros((N, self.d_model, self.d_state), device=tokens.device)
        
        # PRE-COMPUTE DISCRETIZATION for efficiency
        # tokens shape: [N, S, D] -> deltas shape: [N, S, D]
        deltas = self.proj_Delta(tokens)
        Bs = self.proj_B(tokens) # [N, S, d_state]
        Cs = self.proj_C(tokens) # [N, S, d_state]
        
        A_cont = -torch.exp(self.log_A) # [D, d_state]
        
        # Vectorized discretization over the entire sequence
        # A_disc: [N, S, D, d_state]
        A_disc = torch.exp(einsum(deltas, A_cont, 'n s d, d state -> n s d state'))
        # B_disc: [N, S, D, d_state]
        B_disc = einsum(deltas, Bs, 'n s d, n s state -> n s d state')
        
        out_seq = []
        for k in range(S):
            x_k = tokens[:, k, :] # [N, D]
            
            # State Update: h(t) = A * h(t-1) + B * x(t)
            state = A_disc[:, k, :, :] * state + B_disc[:, k, :, :] * rearrange(x_k, 'n d -> n d 1')
            
            # Output: y(t) = C * h(t)
            y_k = einsum(state, Cs[:, k, :], 'n d state, n state -> n d')
            out_seq.append(y_k)
            
        return torch.stack(out_seq, dim=1) # [N, S, D]

    def forward(self, tokens: torch.Tensor):
        # Forward
        out_fwd = self.selective_scan(tokens)
        # Backward
        tokens_rev = torch.flip(tokens, dims=[1])
        out_bwd = self.selective_scan(tokens_rev)
        
        # Sum and mean-pool
        out = out_fwd + torch.flip(out_bwd, dims=[1])
        return out.mean(dim=1) 

class GraphMambaLayer(nn.Module):
    """
    The full residual block: Tokenize -> Scan -> Gated Residual -> LayerNorm
    """
    def __init__(self, d_model: int, seq_len: int = 3): # Reduced seq_len for efficiency
        super().__init__()
        self.d_model = d_model
        
        self.tokenizer = RelationAwareTokenizer(seq_len=seq_len)
        self.ssm_scan = PureBiMambaScan(d_model=d_model)
        
        self.W_g = nn.Linear(d_model, d_model)
        self.W_out = nn.Linear(d_model, d_model)
        self.act = nn.SiLU()
        self.norm = nn.LayerNorm(d_model) # The crucial stability fix

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor, rel_emb_layer: nn.Embedding):
        """
        x: Time-aware node embeddings [N, D]
        edge_index: Graph topology [2, E]
        edge_type: Relation types [E]
        rel_emb_layer: The relation nn.Embedding module from CP-2
        """
        # 1. Tokenize into [N, S, D] (Now requires edge_type and rel_emb)
        tokens = self.tokenizer(x, edge_index, edge_type, rel_emb_layer)
        
        # 2. Bidirectional Scan resulting in [N, D]
        scan_out = self.ssm_scan(tokens)
        
        # 3. Gated Update & Normalization
        gate = self.act(self.W_g(x))
        residual = x + gate * self.W_out(scan_out)
        
        return self.norm(residual)

# ==========================================
# CP-3 VERIFICATION TEST (REFINED)
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("====== CP-3 VERIFICATION (REFINED GRAPH MAMBA) ======")
    print("="*50)
    
    N = 12498
    D = 200
    SEQ_LEN = 3 # Opting for high efficiency
    NUM_REL = 520
    E = 3525
    
    # Dummy Layer Inputs
    x_dummy = torch.randn(N, D)
    src = torch.randint(0, N, (E,))
    dst = torch.randint(0, N, (E,))
    edge_index_dummy = torch.stack([src, dst], dim=0)
    edge_type_dummy = torch.randint(0, NUM_REL, (E,))
    dummy_rel_emb = nn.Embedding(NUM_REL, D)
    
    model = GraphMambaLayer(d_model=D, seq_len=SEQ_LEN)
    
    out = model(x_dummy, edge_index_dummy, edge_type_dummy, dummy_rel_emb)
    
    print(f"Final Mamba Output Shape: {list(out.shape)} (Expected: [{N}, 200])")
    print(f"Requires Grad: {out.requires_grad}")
    if torch.isnan(out).any():
        print("-> [FAIL] NaN detected in output! LayerNorm failed.")
    else:
        print("-> [PASS] Output is numerically stable.")
    print("="*50 + "\n")