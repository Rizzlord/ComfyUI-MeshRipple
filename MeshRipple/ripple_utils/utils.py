
import torch
import os
import re

def count_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    param_size_bytes = total_params * 4
    param_size_mb = param_size_bytes / (1024 ** 2)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Estimated size: {param_size_mb:.2f} MB")

    return total_params, trainable_params, param_size_mb

def update_eos_mask(next_output, eos_mask, token_map):
    device = next_output.device
    eos_token = token_map['eos'].to(device)
    expanded_eos_token = eos_token.unsqueeze(0).expand_as(next_output)
    new_eos_mask = torch.all(next_output == expanded_eos_token, dim=-1) 
    new_eos_mask = new_eos_mask | eos_mask
    return new_eos_mask