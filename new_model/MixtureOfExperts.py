import torch.nn as nn
from GatedFeedForward import GatedFeedForward
import torch
import torch.nn.functional as F
class MixtureOfExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_moe_experts
        self.top_k = config.moe_top_k
        self.experts = nn.ModuleList([GatedFeedForward(config) for _ in range(self.num_experts)])
        self.gate = nn.Linear(config.hidden_size, self.num_experts)

    def forward(self, x):
        expert_weights = F.softmax(self.gate(x), dim=-1)
        top_k_weights, top_k_indices = torch.topk(expert_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = top_k_indices == i
            if mask.any():
                expert_input = x[mask]
                expert_output = expert(expert_input)
                output[mask] += top_k_weights[mask][:, i].unsqueeze(-1) * expert_output
        
        return output