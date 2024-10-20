import math
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
class EnhancedDecoderOnlyConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        tie_word_embeddings=True,
        use_moe=False,
        num_experts=4,
        top_k_experts=2,
        use_cognitive_layer=True,
        use_ntk_layer=True,
        use_decision_trees=True,
        learning_rate=5e-5,
        max_grad_norm=1.0,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        label_smoothing=0.1,
        **kwargs
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.use_cognitive_layer = use_cognitive_layer
        self.use_ntk_layer = use_ntk_layer
        self.use_decision_trees = use_decision_trees
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_ratio = warmup_ratio
        self.label_smoothing = label_smoothing

class CognitiveLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.n_embd
        
        # Working Memory
        self.working_memory = nn.Linear(config.n_embd, config.n_embd)
        self.memory_gate = nn.Linear(2 * config.n_embd, config.n_embd)
        
        # Long-term Memory (GRU)
        self.gru = nn.GRU(config.n_embd, config.n_embd, batch_first=True)
        
        # Attention Control
        self.attention_control = nn.MultiheadAttention(config.n_embd, config.n_head, batch_first=True)
        
    def forward(self, hidden_states, memory_state=None):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Working Memory
        working_mem = self.working_memory(hidden_states)
        
        # Long-term Memory
        if memory_state is None:
            memory_state = torch.zeros(1, batch_size, self.hidden_size, device=hidden_states.device)
        long_term_mem, new_memory_state = self.gru(hidden_states, memory_state)
        
        # Combine memories
        gate = torch.sigmoid(self.memory_gate(torch.cat([working_mem, long_term_mem], dim=-1)))
        combined_mem = gate * working_mem + (1 - gate) * long_term_mem
        
        # Attention Control
        attended_mem, _ = self.attention_control(combined_mem, combined_mem, combined_mem)
        
        return attended_mem, new_memory_state

class DynamicNTKLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feature_map = nn.Linear(config.n_embd, config.n_embd * 2)
        self.output_map = nn.Linear(config.n_embd * 2, config.n_embd)
        
    def forward(self, x):
        features = self.feature_map(x)
        features = F.relu(features) ** 2
        return self.output_map(features)

class DecisionTree(nn.Module):
    def __init__(self, input_dim, depth):
        super().__init__()
        self.depth = depth
        self.num_leaves = 2 ** depth
        self.num_decision_nodes = self.num_leaves - 1
        
        self.decision_nodes = nn.ModuleList([
            nn.Linear(input_dim, 1) for _ in range(self.num_decision_nodes)
        ])
        self.leaf_nodes = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(self.num_leaves)
        ])
    
    def _compute_leaf_probabilities(self, x):
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Initialize probabilities for all nodes (decision nodes + leaves)
        all_probs = torch.ones(batch_size, seq_len, 2 * self.num_leaves - 1, device=device)
        
        # Compute decision probabilities for all internal nodes
        flat_x = x.view(-1, x.size(-1))
        
        for node in range(self.num_decision_nodes):
            decision = torch.sigmoid(self.decision_nodes[node](flat_x)).view(batch_size, seq_len)
            
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            
            # Ensure we're not exceeding tensor bounds
            if left_child < all_probs.size(2):
                all_probs[:, :, left_child] *= decision
            if right_child < all_probs.size(2):
                all_probs[:, :, right_child] *= (1 - decision)
        
        # Extract leaf probabilities (last num_leaves entries)
        leaf_probabilities = all_probs[:, :, self.num_decision_nodes:]
        
        return leaf_probabilities
    
    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        device = x.device
        
        # Compute leaf probabilities
        leaf_probabilities = self._compute_leaf_probabilities(x)
        
        # Apply leaf transformations
        leaf_outputs = []
        for leaf in self.leaf_nodes:
            leaf_output = leaf(x)  # Shape: [batch, seq, input_dim]
            leaf_outputs.append(leaf_output)
        
        # Stack leaf outputs: [batch, seq, num_leaves, input_dim]
        leaf_outputs = torch.stack(leaf_outputs, dim=2)
        
        # Apply leaf probabilities
        weighted_outputs = leaf_outputs * leaf_probabilities.unsqueeze(-1)
        
        # Sum over leaves
        final_output = weighted_outputs.sum(dim=2)
        
        return final_output
    
class DecoderOnlyAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.attn_pdrop

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _attn(self, query, key, value, attention_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        attn_weights = attn_weights / torch.sqrt(torch.tensor(value.size(-1), dtype=torch.float32))
        
        if attention_mask is not None:
            # Ensure the attention_mask has the right shape
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights

    def forward(self, hidden_states, attention_mask=None, use_cache=False, past_key_value=None):
        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(self.n_embd, dim=2)
        
        query = query.view(*query.size()[:-1], self.n_head, -1).transpose(1, 2)
        key = key.view(*key.size()[:-1], self.n_head, -1).transpose(1, 2)
        value = value.view(*value.size()[:-1], self.n_head, -1).transpose(1, 2)
        
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)
        
        attn_output, attn_weights = self._attn(query, key, value, attention_mask)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(*hidden_states.size()[:-1], self.n_embd)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        outputs = (attn_output,)
        
        if use_cache:
            outputs += ((key, value),)
            
        return outputs

class DecoderOnlyMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_inner)
        self.c_proj = nn.Linear(config.n_inner, config.n_embd)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class ParallelDecisionTrees(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_trees = 2  # Reduced from 4 to 2 for efficiency
        self.depth = 3      # Reduced from 3 to 2 for efficiency
        
        self.trees = nn.ModuleList([
            DecisionTree(config.n_embd, depth=self.depth) for _ in range(self.num_trees)
        ])
        self.combiner = nn.Linear(config.n_embd * self.num_trees, config.n_embd)
        
    def forward(self, x):
        tree_outputs = [tree(x) for tree in self.trees]
        combined = torch.cat(tree_outputs, dim=-1)
        return self.combiner(combined)

class MixtureOfExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        self.experts = nn.ModuleList([DecoderOnlyMLP(config) for _ in range(self.num_experts)])
        self.gate = nn.Linear(config.n_embd, self.num_experts)

    def forward(self, hidden_states):
        expert_weights = F.softmax(self.gate(hidden_states), dim=-1)
        top_k_weights, top_k_indices = torch.topk(expert_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        output = torch.zeros_like(hidden_states)
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, :, k]
            expert_weight = top_k_weights[:, :, k].unsqueeze(-1)
            for i in range(self.num_experts):
                expert_mask = (expert_idx == i)
                if expert_mask.any():
                    expert_input = hidden_states[expert_mask]
                    expert_output = self.experts[i](expert_input)
                    output[expert_mask] += expert_weight[expert_mask] * expert_output
        
        return output
    
# Update the DecoderOnlyBlock class
class DecoderOnlyBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = DecoderOnlyAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
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
        
        attn_outputs = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_value=past_key_value
        )
        
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
            
        return outputs

class EnhancedDecoderOnlyGPT(PreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([EnhancedDecoderOnlyBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        if config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            self.lm_head.weight = self.wte.weight
        else:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            
        self.init_weights()

    def init_weights(self):
        # Initialize weights
        for name, param in self.named_parameters():
            if "weight" in name:
                if param.dim() >= 2:  # For weights in layers (>= 2D)
                    if "attn" in name:  # Multi-head attention layers
                        # Orthogonal initialization for attention weights
                        nn.init.orthogonal_(param, gain=math.sqrt(2))  # Use a scaling factor for ReLU
                    elif "ffn" in name or "linear" in name:  # Feed-forward layers and output layer
                        # Kaiming initialization for feed-forward layers
                        nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
                    else:  # General layer initialization
                        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(param)
                        bound = 1 / math.sqrt(fan_in)
                        nn.init.uniform_(param, -bound, bound)
                else:  # For 1D tensors (e.g., embeddings)
                    nn.init.normal_(param, mean=0, std=0.01)
            elif "bias" in name:
                # Small positive bias initialization
                nn.init.normal_(param, mean=0.01, std=0.02)  # Helps with learning

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings


    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        past_key_values=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        
        for i, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            past_key_value = past_key_values[i] if past_key_values is not None else None

            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = outputs[0]
            if use_cache:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + (presents,) + (hidden_states,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

class EnhancedDecoderOnlyBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = DecoderOnlyAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
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
            
    def forward(self, hidden_states, attention_mask=None, past_key_value=None, use_cache=False, output_attentions=False):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        
        attn_outputs = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_value=past_key_value
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        
        hidden_states = residual + attn_output
        
        if hasattr(self, 'cognitive'):
            cognitive_output, _ = self.cognitive(hidden_states)
            hidden_states = hidden_states + cognitive_output
            
        if hasattr(self, 'ntk'):
            ntk_output = self.ntk(hidden_states)
            hidden_states = hidden_states + ntk_output
            
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states
        
        if hasattr(self, 'decision_trees'):
            tree_output = self.decision_trees(hidden_states)
            hidden_states = hidden_states + tree_output
            
        outputs = (hidden_states,) + outputs
        return outputs

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path
import logging
import wandb
from tqdm.auto import tqdm
import json
from dataclasses import dataclass
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import random

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_steps: int = -1
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 1000
    save_total_limit: int = 5
    fp16: bool = False
    fp16_opt_level: str = "O2"
    local_rank: int = -1
    seed: int = 42
    output_dir: str = "checkpoints"

from datasets import load_dataset
from torch.utils.data import Dataset
import torch
from transformers import GPT2Tokenizer
from typing import Optional, Dict
import re
from pathlib import Path



class WikipediaDataset(Dataset):
    def __init__(
        self,
        tokenizer: GPT2Tokenizer,
        max_length: int = 512,
        cache_dir: Optional[str] = None,
        num_proc: int = 4,
        streaming: bool = False,
        split: str = "train[:2000]",
        load_from_cache_file: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Load dataset with proper caching and streaming options
        self.dataset = load_dataset(
            "wikipedia",
            "20220301.en",
            split=split,
            streaming=streaming,
            cache_dir=cache_dir
        )
        
        # Apply preprocessing using map
        self.processed_dataset = self.dataset.map(
            self._preprocess_function,
            remove_columns=self.dataset.column_names,
            num_proc=num_proc if not streaming else None,
            load_from_cache_file=load_from_cache_file,
            desc="Processing Wikipedia articles"
        )
        
        # Filter out empty examples
        self.processed_dataset = self.processed_dataset.filter(
            lambda x: len(x['input_ids']) > 0
        )

    def _preprocess_function(self, examples: Dict) -> Dict:
        """Preprocesses the Wikipedia examples."""
        # Clean text
        text = self._clean_text(examples['text'])
        
        # Skip empty texts
        if not text:
            return {'input_ids': [], 'attention_mask': [], 'labels': []}

        # Tokenize
        tokenized = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Create labels for language modeling
        labels = tokenized['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }

    @staticmethod
    def _clean_text(text: str) -> str:
        """Cleans Wikipedia text."""
        # Remove references
        text = re.sub(r'\[\d+\]', '', text)
        
        # Remove parenthetical text
        text = re.sub(r'\s*\([^)]*\)', '', text)
        
        # Remove multiple spaces and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        return text.strip()

    def __len__(self) -> int:
        return len(self.processed_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.processed_dataset[idx]
        if len(item['input_ids']) == 0:
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'labels': torch.full((self.max_length,), -100, dtype=torch.long)  # Padding for labels
            }
        return item

def collate_fn(batch):
    def to_tensor(item):
        if isinstance(item, torch.Tensor):
            return item
        elif isinstance(item, list):
            return torch.tensor(item)
        else:
            raise TypeError(f"Unsupported type: {type(item)}")

    batch = [item for item in batch if len(item['input_ids']) > 0]  # Filter out empty items
    if not batch:  # If the batch is empty after filtering
        return {
            'input_ids': torch.zeros(0, dtype=torch.long),
            'attention_mask': torch.zeros(0, dtype=torch.long),
            'labels': torch.full((0,), -100, dtype=torch.long)
        }

    input_ids = torch.stack([to_tensor(item['input_ids']) for item in batch])
    attention_mask = torch.stack([to_tensor(item['attention_mask']) for item in batch])
    labels = torch.stack([to_tensor(item['labels']) for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def train_enhanced_chatbot_model(
    model: nn.Module,
    tokenizer: GPT2Tokenizer,
    train_dataset: WikipediaDataset,
    val_dataset: WikipediaDataset,  # Add validation dataset
    config: TrainingConfig,
    start_epoch: int = 0,
    train_loss_history: Optional[List[float]] = None
) -> Tuple[nn.Module, List[float]]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    # Initialize wandb for experiment tracking
    if config.local_rank in [-1, 0]:
        wandb.init(project="enhanced-chatbot", config=vars(config))
    
    # Setup distributed training
    if config.local_rank != -1:
        torch.cuda.set_device(config.local_rank)
        device = torch.device("cuda", config.local_rank)
        dist.init_process_group(backend='nccl')
        model = DDP(model, device_ids=[config.local_rank])
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)

    train_sampler = DistributedSampler(train_dataset) if config.local_rank != -1 else None
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    
    # Validation DataLoader
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn
    )

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': config.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.learning_rate,
        eps=config.adam_epsilon
    )

    num_update_steps_per_epoch = len(train_dataloader) // config.gradient_accumulation_steps
    if config.max_steps > 0:
        total_steps = config.max_steps
        num_train_epochs = config.max_steps // num_update_steps_per_epoch
    else:
        total_steps = num_update_steps_per_epoch * config.num_epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps
    )

    scaler = GradScaler() if config.fp16 else None

    if train_loss_history is None:
        train_loss_history = []

    logger.info("Starting training...")
    global_step = 0
    
    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        # Initialize metrics for the epoch
        running_loss = 0.0
        running_step_loss = 0.0
        steps_since_last_log = 0
        
        epoch_iterator = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch}",
            disable=config.local_rank not in [-1, 0],
            position=0,
            leave=True
        )
        
        for step, batch in enumerate(epoch_iterator):
            # Handle the batch format based on your dataset structure
            if isinstance(batch, list):
                inputs = {
                    'input_ids': batch[0].to(device),
                    'attention_mask': batch[1].to(device) if len(batch) > 1 else None,
                    'labels': batch[2].to(device) if len(batch) > 2 else None
                }
                inputs = {k: v for k, v in inputs.items() if v is not None}
            else:
                inputs = {k: v.to(device) for k, v in batch.items()}

            with autocast('cuda', enabled=config.fp16):
                outputs = model(**inputs)
                loss = outputs.loss / config.gradient_accumulation_steps

            if torch.isnan(loss):
                optimizer.zero_grad()  # Reset gradients
                print("nan")
                continue
                
            if config.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update metrics
            running_loss += loss.item()
            running_step_loss += loss.item()
            steps_since_last_log += 1
            train_loss_history.append(loss.item())

            # Calculate average losses
            avg_loss = running_loss / (step + 1)
            step_avg_loss = running_step_loss / steps_since_last_log
            
            # Update tqdm description with metrics
            epoch_iterator.set_description(
                f"Epoch {epoch} | "
                f"AVGLoss: {avg_loss:.8f} | "
                f"Step Loss: {step_avg_loss:.8f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                f"Steps: {global_step}"
            )

            if config.local_rank in [-1, 0] and step % config.logging_steps == 0:
                wandb.log({
                    'loss': loss.item(),
                    'avg_loss': avg_loss,
                    'step_loss': step_avg_loss,
                    'learning_rate': scheduler.get_last_lr()[0],
                    'epoch': epoch,
                    'step': global_step
                })
                # Reset step loss tracking
                running_step_loss = 0.0
                steps_since_last_log = 0

            if (step + 1) % config.gradient_accumulation_steps == 0:
                if config.fp16:
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.max_grad_norm
                )

                if config.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                    
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if (config.local_rank in [-1, 0] and 
                    config.save_steps > 0 and 
                    global_step % config.save_steps == 0):
                    save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        config,
                        epoch,
                        global_step,
                        train_loss_history
                    )
            
            if config.max_steps > 0 and global_step >= config.max_steps:
                epoch_iterator.close()
                break
        
        if config.max_steps > 0 and global_step >= config.max_steps:
            break
        
        # Validation Phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_dataloader:
                if isinstance(val_batch, list):
                    inputs = {
                        'input_ids': val_batch[0].to(device),
                        'attention_mask': val_batch[1].to(device) if len(val_batch) > 1 else None,
                        'labels': val_batch[2].to(device) if len(val_batch) > 2 else None
                    }
                    inputs = {k: v for k, v in inputs.items() if v is not None}
                else:
                    inputs = {k: v.to(device) for k, v in val_batch.items()}

                outputs = model(**inputs)
                val_loss += outputs.loss.item()

            avg_val_loss = val_loss / len(val_dataloader)

        # Log validation metrics
        if config.local_rank in [-1, 0]:
            wandb.log({
                'val_loss': avg_val_loss,
                'epoch': epoch,
                'step': global_step
            })
            logger.info(f"Validation Loss: {avg_val_loss:.4f}")

    # Final save
    if config.local_rank in [-1, 0]:
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            config,
            epoch,
            global_step,
            train_loss_history,
            is_final=True
        )
        
    return model, train_loss_history


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    config: TrainingConfig,
    epoch: int,
    global_step: int,
    train_loss_history: List[float],
    is_final: bool = False
) -> None:
    """Save training checkpoint"""
    output_dir = Path(config.output_dir)
    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    if isinstance(model, DDP):
        model = model.module
    
    model.save_pretrained(checkpoint_dir,safe_serialization=False)
    
    # Save training state
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss_history': train_loss_history
    }, checkpoint_dir / "training_state.pt")
    with open(checkpoint_dir / "training_config.json", 'w') as f:
        json.dump(vars(config), f, indent=4)
    if config.save_total_limit > 0:
        cleanup_checkpoints(output_dir, config.save_total_limit, 
                          exclude=checkpoint_dir if is_final else None)
    logger.info(f"Saved checkpoint: {checkpoint_dir}")

def cleanup_checkpoints(
    output_dir: Path,
    save_total_limit: int,
    exclude: Optional[Path] = None
) -> None:
    checkpoints = sorted(
        [d for d in output_dir.glob("checkpoint-*") if d != exclude],
        key=lambda x: int(x.name.split('-')[1])
    )
    
    if len(checkpoints) > save_total_limit:
        for checkpoint in checkpoints[:-save_total_limit]:
            logger.info(f"Deleting old checkpoint: {checkpoint}")
            shutil.rmtree(checkpoint)


def main(num_examples=None, resume_from=None):
    training_config = TrainingConfig(
        batch_size= 1,
        num_epochs= 3,
        learning_rate= 1e-4,
    )
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2",clean_up_tokenization_spaces=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Initializing new model")
    config = EnhancedDecoderOnlyConfig(
        vocab_size=tokenizer.vocab_size,
        n_positions=512,
        n_embd=512,
        n_layer=6,
        n_head=8,
        n_inner=768*4,
        learning_rate=training_config.learning_rate, 
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        use_moe=True,
        num_experts=2,
        top_k_experts=2,
        use_cognitive_layer=True,
        use_ntk_layer=True,
        use_decision_trees=True,
        num_epochs=training_config.num_epochs,
        batch_size=training_config.batch_size,
        warmup_steps=100,
        seed =training_config.seed,
        local_rank =training_config.local_rank,
        weight_decay=training_config.weight_decay,
        adam_epsilon= training_config.adam_epsilon,
        max_steps = training_config.max_steps,
        fp16=training_config.fp16,
        logging_steps =training_config.logging_steps,
        save_steps=training_config.save_steps,
        output_dir=training_config.output_dir,
        save_total_limit = training_config.save_total_limit
    )
    model = EnhancedDecoderOnlyGPT(config)
    start_epoch = 0
    train_loss_history = None

    # Create dataset
    dataset = WikipediaDataset(
            tokenizer=tokenizer,
            max_length=512,
            cache_dir="./cache",
            num_proc=4
        )

    # Train the model
    model, loss_history = train_enhanced_chatbot_model(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            config=config
        )
    return model, tokenizer, train_loss_history

if __name__ == "__main__":
    trained_model, tokenizer, loss_history = main(
        num_examples=100,
    )