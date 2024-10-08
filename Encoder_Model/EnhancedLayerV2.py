from EnhancedAttentionV2 import EnhancedAttentionV2
from GatedFeedForward import GatedFeedForward
import torch.nn as nn
from DecisionTreeLayer import DecisionTreeLayer
from ThoughtLayer import ThoughtLayer
from Adapter import Adapter
from MixtureOfExperts import MixtureOfExperts
from DynamicNTKLayer import DynamicNTKLayer

class EnhancedLayerV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = EnhancedAttentionV2(config)
        self.ffn = GatedFeedForward(config) if config.use_gated_ffn else nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.tree = DecisionTreeLayer(config)
        self.thought = ThoughtLayer(config)
        if config.use_adapter:
            self.adapter = Adapter(config)
        if config.use_mixture_of_experts:
            self.moe = MixtureOfExperts(config)
        if config.use_dynamic_ntk:
            self.dynamic_ntk = DynamicNTKLayer(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        ffn_output = self.ffn(attention_output)
        tree_output = self.tree(ffn_output)
        thought_output = self.thought(tree_output)
        
        if hasattr(self, 'adapter'):
            thought_output = self.adapter(thought_output)
        
        if hasattr(self, 'moe'):
            thought_output = self.moe(thought_output)
        
        if hasattr(self, 'dynamic_ntk'):
            thought_output = self.dynamic_ntk(thought_output)
        
        return self.layer_norm(hidden_states + thought_output)