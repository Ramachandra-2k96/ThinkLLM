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
        # x shape: [batch_size, seq_len, hidden_size]
        batch_size, seq_len, hidden_size = x.shape
        
        # Compute expert weights
        expert_weights = F.softmax(self.gate(x), dim=-1)
        # expert_weights shape: [batch_size, seq_len, num_experts]
        
        # Get top-k expert weights and indices
        top_k_weights, top_k_indices = torch.topk(expert_weights, self.top_k, dim=-1)
        # top_k_weights shape: [batch_size, seq_len, top_k]
        # top_k_indices shape: [batch_size, seq_len, top_k]
        
        # Normalize top-k weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # Initialize output tensor
        output = torch.zeros_like(x)
        
        # Apply each expert
        for i, expert in enumerate(self.experts):
            # Create a mask for the current expert
            mask = (top_k_indices == i).any(dim=-1)  # shape: [batch_size, seq_len]
            
            if mask.any():
                # Select the inputs for this expert
                expert_input = x[mask]
                
                # Apply the expert
                expert_output = expert(expert_input)
                
                # Get the weights for this expert
                expert_weights = top_k_weights[mask][top_k_indices[mask] == i]
                
                # Add the weighted output to the result
                output[mask] += expert_weights.unsqueeze(-1) * expert_output
        
        return output