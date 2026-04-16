import torch
import torch.nn as nn
import math

class AsymmetricNoiseScheduler(nn.Module):
    """
    Stage 6a & 6b: Asymmetric Forward Diffusion & DDPM Reverse Sampling.
    """
    def __init__(self, num_timesteps=100, beta_start=1e-4, beta_end=0.02, elevated_beta=5.0):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.elevated_beta = elevated_beta
        
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # --- DDPM REVERSE SAMPLING BUFFERS ---
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        # Shift cumprod by 1 and pad with 1.0 for the t-1 step
        self.register_buffer('alphas_cumprod_prev', torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]]))
        
        # --- FORWARD NOISE BUFFERS ---
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    def forward_noise(self, h_0, m, branch_type="causal"):
        """Adds forward noise to the representation."""
        noise = torch.randn_like(h_0)
        
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[m].view(-1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[m].view(-1, 1)
        
        if branch_type == "causal":
            h_m = sqrt_alpha_bar * h_0 + sqrt_one_minus_alpha_bar * noise
        else:
            elevated_variance = math.sqrt(self.elevated_beta) * sqrt_one_minus_alpha_bar
            h_m = sqrt_alpha_bar * h_0 + elevated_variance * noise
            
        return h_m, noise

    def reverse_step(self, x_t, x_0_pred, t):
        """
        The DDPM Posterior Step (t -> t-1).
        """
        beta_t = self.betas[t].view(-1, 1)
        alpha_t = self.alphas[t].view(-1, 1)
        alpha_bar_t = self.alphas_cumprod[t].view(-1, 1)
        alpha_bar_t_prev = self.alphas_cumprod_prev[t].view(-1, 1)

        coef1 = (beta_t * torch.sqrt(alpha_bar_t_prev)) / (1.0 - alpha_bar_t)
        coef2 = ((1.0 - alpha_bar_t_prev) * torch.sqrt(alpha_t)) / (1.0 - alpha_bar_t)
        mean = coef1 * x_0_pred + coef2 * x_t

        var = beta_t * (1.0 - alpha_bar_t_prev) / (1.0 - alpha_bar_t)
        var = torch.clamp(var, min=1e-20) 

        noise = torch.randn_like(x_t)
        nonzero_mask = (t > 0).float().view(-1, 1) 

        x_t_minus_1 = mean + nonzero_mask * torch.sqrt(var) * noise
        return x_t_minus_1


class BiSSMDenoiser(nn.Module):
    """
    Stage 6c: Bidirectional SSM Denoiser.
    Dynamically scaled (Multi-layer support + Dropout).
    """
    def __init__(self, d_model: int, d_state: int = 16, num_timesteps: int = 100, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        
        self.step_embed = nn.Embedding(num_timesteps, d_model)
        self.s_proj = nn.Linear(d_model * d_state, d_model)
        
        self.condition_fusion = nn.Sequential(
            nn.Linear(d_model * 5, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        self.bi_ssm_scan = nn.GRU(
            input_size=d_model, 
            hidden_size=d_model // 2, 
            num_layers=num_layers,        # Dynamic depth mapping
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, h_noisy, r_emb, s_T_raw, t_emb, m):
        emb_m = self.step_embed(m.long()) 
        
        s_T_flat = s_T_raw.view(s_T_raw.size(0), -1) 
        s_T_proj = self.s_proj(s_T_flat)             
        
        cond_input = torch.cat([h_noisy, r_emb, s_T_proj, emb_m, t_emb], dim=-1)
        e_m = self.condition_fusion(cond_input) 
        
        seq_input = e_m.unsqueeze(0) 
        scanned_out, _ = self.bi_ssm_scan(seq_input)
        scanned_out = scanned_out.squeeze(0)
        
        return self.out_norm(scanned_out + h_noisy)