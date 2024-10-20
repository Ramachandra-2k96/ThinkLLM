import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticMemoryModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_concepts = 1000
        self.concept_dim = config.n_embd
        
        # Learnable concept embeddings
        self.concept_embeddings = nn.Parameter(
            torch.randn(self.num_concepts, self.concept_dim)
        )
        
        self.concept_projector = nn.Linear(config.n_embd, self.concept_dim)
        self.output_projector = nn.Linear(self.concept_dim, config.n_embd)
        
    def forward(self, hidden_states):
        # Project hidden states to concept space
        projected_states = self.concept_projector(hidden_states)
        
        # Compute attention with concepts
        attention_scores = torch.matmul(projected_states, self.concept_embeddings.t())
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Weighted sum of concepts
        concept_mixture = torch.matmul(attention_probs, self.concept_embeddings)
        
        # Project back to hidden space
        return self.output_projector(concept_mixture)