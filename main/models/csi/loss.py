import torch.nn.functional as F
import torch

def csi_loss(pred_c, pred_s, pred_i, target,
             lambda_unif=0.1, lambda_causal=0.1):

    loss_sup = F.cross_entropy(pred_c, target)

    uniform = torch.full_like(pred_s, 1.0 / pred_s.size(1))
    loss_unif = F.kl_div(F.log_softmax(pred_s, dim=-1),
                         uniform,
                         reduction="batchmean")

    loss_causal = F.cross_entropy(pred_i, target)

    return loss_sup + lambda_unif * loss_unif + lambda_causal * loss_causal