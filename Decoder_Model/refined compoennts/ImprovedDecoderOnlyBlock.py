import torch
import torch.nn as nn
from DecoderOnlyAttention import DecoderOnlyAttention
from HierarchicalCompression import HierarchicalCompression
from SemanticMemoryModule import SemanticMemoryModule
from MixtureOfExperts import MixtureOfExperts
from DecoderOnlyMLP import DecoderOnlyMLP
from CognitiveLayer import CognitiveLayer
from DynamicNTKLayer import DynamicNTKLayer
from ParallelDecisionTrees import ParallelDecisionTrees

class ImprovedDecoderOnlyBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = DecoderOnlyAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # Enhanced components
        self.hierarchical_comp = HierarchicalCompression(config)
        self.semantic_memory = SemanticMemoryModule(config)
        
        if config.use_moe:
            self.mlp = MixtureOfExperts(config)
        else:
            self.mlp = DecoderOnlyMLP(config)
            
        if config.use_cognitive_layer:
            self.cognitive = CognitiveLayer(config)
        
        if config.use_ntk_layer:
            self.ntk = DynamicNTKLayer(config)
            
        if config.use_decision_trees:
            self.decision_trees = ParallelDecisionTrees(config)
            
    def forward(self, hidden_states, attention_mask=None, memory_state=None, use_cache=False, past_key_value=None):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        
        # Apply hierarchical compression
        compressed_states = self.hierarchical_comp(hidden_states)
        hidden_states = hidden_states + compressed_states
        
        # Apply semantic memory
        semantic_states = self.semantic_memory(hidden_states)
        hidden_states = hidden_states + semantic_states
        
        attn_outputs = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_value=past_key_value
        )
        
        # Rest of the forward pass remains the same
        if isinstance(attn_outputs, tuple):
            attn_output = attn_outputs[0]
            present_key_value = attn_outputs[1] if len(attn_outputs) > 1 else None
        else:
            attn_output = attn_outputs
            present_key_value = None
            
        hidden_states = residual + attn_output
        
        new_memory_state = None
        if hasattr(self, 'cognitive'):
            cognitive_output, new_memory_state = self.cognitive(hidden_states, memory_state)
            hidden_states = hidden_states + cognitive_output
            
        if hasattr(self, 'ntk'):
            ntk_output = self.ntk(hidden_states)
            hidden_states = hidden_states + ntk_output
            
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output
        
        if hasattr(self, 'decision_trees'):
            tree_output = self.decision_trees(hidden_states)
            hidden_states = hidden_states + tree_output
            
        outputs = (hidden_states,)
        
        if new_memory_state is not None:
            outputs += (new_memory_state,)
            
        if use_cache:
            outputs += (present_key_value,)
            
        return hidden_states, new_memory_state, present_key_value if use_cache else None
