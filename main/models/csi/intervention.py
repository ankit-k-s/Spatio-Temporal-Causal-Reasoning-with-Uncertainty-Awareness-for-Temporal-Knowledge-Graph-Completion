import torch

def intervene(x):
    idx = torch.randperm(x.size(0))
    return x[idx]