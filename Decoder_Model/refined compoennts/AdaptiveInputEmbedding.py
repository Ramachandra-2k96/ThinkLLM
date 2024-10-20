import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveInputEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.adaptive_span = nn.Parameter(torch.ones(config.vocab_size))
        self.base_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.context_projector = nn.Linear(config.n_embd, config.n_embd)
        
    def forward(self, input_ids):
        # Get base embeddings
        base_embeds = self.base_embedding(input_ids)
        
        # Apply adaptive span attention
        span_weights = F.softmax(self.adaptive_span, dim=0)
        span_weights = span_weights[input_ids].unsqueeze(-1)
        
        # Project embeddings based on context
        context_embeds = self.context_projector(base_embeds)
        
        # Combine adaptively
        return span_weights * context_embeds + (1 - span_weights) * base_embeds