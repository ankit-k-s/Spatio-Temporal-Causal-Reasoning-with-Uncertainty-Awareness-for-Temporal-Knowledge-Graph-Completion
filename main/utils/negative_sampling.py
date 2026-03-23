import torch

def sample_negatives(num_entities, batch_size, num_neg=10):
    return torch.randint(0, num_entities, (batch_size, num_neg))