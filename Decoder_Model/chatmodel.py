import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from tqdm.auto import tqdm
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

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

    def forward(self, hidden_states, attention_mask=None):
        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(self.n_embd, dim=2)
        
        query = query.view(*query.size()[:-1], self.n_head, -1).transpose(1, 2)
        key = key.view(*key.size()[:-1], self.n_head, -1).transpose(1, 2)
        value = value.view(*value.size()[:-1], self.n_head, -1).transpose(1, 2)

        attn_output, _ = self._attn(query, key, value, attention_mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(*hidden_states.size()[:-1], self.n_embd)
        
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        return attn_output

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
        self.depth = 2      # Reduced from 3 to 2 for efficiency
        
        self.trees = nn.ModuleList([
            DecisionTree(config.n_embd, depth=self.depth) for _ in range(self.num_trees)
        ])
        self.combiner = nn.Linear(config.n_embd * self.num_trees, config.n_embd)
        
    def forward(self, x):
        tree_outputs = [tree(x) for tree in self.trees]
        combined = torch.cat(tree_outputs, dim=-1)
        return self.combiner(combined)

class DecoderOnlyConfig(PretrainedConfig):
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
        
        # New stability parameters
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_ratio = warmup_ratio

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
            
    def forward(self, hidden_states, attention_mask=None, memory_state=None):
        attn_output = self.attn(self.ln_1(hidden_states), attention_mask=attention_mask)
        hidden_states = hidden_states + attn_output
        
        if hasattr(self, 'cognitive'):
            cognitive_output, new_memory_state = self.cognitive(hidden_states, memory_state)
            hidden_states = hidden_states + cognitive_output
        else:
            new_memory_state = None
            
        if hasattr(self, 'ntk'):
            ntk_output = self.ntk(hidden_states)
            hidden_states = hidden_states + ntk_output
            
        mlp_output = self.mlp(self.ln_2(hidden_states))
        hidden_states = hidden_states + mlp_output
        
        if hasattr(self, 'decision_trees'):
            tree_output = self.decision_trees(hidden_states)
            hidden_states = hidden_states + tree_output
            
        return hidden_states, new_memory_state

# Update the DecoderOnlyGPT class
class DecoderOnlyGPT(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([DecoderOnlyBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        if config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            self.lm_head.weight = self.wte.weight
        else:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            
        self.init_weights()
        
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
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
            
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])
            
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
            
        if position_ids is None:
            position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=inputs_embeds.device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
            
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds
            
        hidden_states = self.drop(hidden_states)
        
        memory_states = [None] * len(self.h)
        for i, block in enumerate(self.h):
            hidden_states, memory_states[i] = block(hidden_states, attention_mask, memory_states[i])
            
        hidden_states = self.ln_f(hidden_states)
        lm_logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        return {
            "loss": loss,
            "logits": lm_logits,
            "hidden_states": hidden_states,
            "memory_states": memory_states,

        }
    def generate(
        self,
        input_ids,
        max_length=100,
        min_length=10,
        num_beams=4,
        num_beam_groups=1,
        diversity_penalty=0.0,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        length_penalty=1.0,
        pad_token_id=None,
        eos_token_id=None,
        attention_mask=None,
        num_return_sequences=1,
        use_cache=True,
        decoder_strategy="beam_search",  # Options: "beam_search", "greedy", "sample"
        cognitive_memory_influence=0.3,
        expert_confidence_threshold=0.7
    ):
        # Ensure proper device placement
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        batch_size = input_ids.shape[0]
        vocab_size = self.config.vocab_size
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)
        
        class BeamHypotheses:
            def __init__(self, num_beams, length_penalty, early_stopping=False):
                self.max_length = max_length
                self.num_beams = num_beams
                self.length_penalty = length_penalty
                self.early_stopping = early_stopping
                self.beams = []
                self.worst_score = 1e9

            def add(self, hyp, sum_logprobs, cognitive_state=None, expert_usage=None):
                score = sum_logprobs / (len(hyp) ** self.length_penalty)
                if len(self.beams) < self.num_beams or score > self.worst_score:
                    self.beams.append((score, hyp, cognitive_state, expert_usage))
                    if len(self.beams) > self.num_beams:
                        sorted_next_scores = sorted([(s, idx) for idx, (s, _, _, _) in enumerate(self.beams)])
                        del self.beams[sorted_next_scores[0][1]]
                        self.worst_score = sorted_next_scores[1][0]
                    else:
                        self.worst_score = min(score, self.worst_score)

        def get_expert_confidence(logits, expert_history):
            if hasattr(self, 'h') and any(hasattr(layer.mlp, 'gate') for layer in self.h):
                expert_weights = torch.softmax(logits, dim=-1)
                top_expert = torch.argmax(expert_weights).item()
                if expert_history and top_expert in expert_history:
                    return expert_history[top_expert] / sum(expert_history.values())
            return 0.0

        def cognitive_state_similarity(state1, state2):
            if state1 is None or state2 is None:
                return 0.0
            return F.cosine_similarity(state1.unsqueeze(0), state2.unsqueeze(0)).item()

        # Initialize beam hypotheses for each batch
        generated_hyps = [
            BeamHypotheses(num_beams, length_penalty) 
            for _ in range(batch_size)
        ]
        
        # Expand input for beam search
        input_ids = input_ids.unsqueeze(1).expand(batch_size, num_beams, -1)
        input_ids = input_ids.contiguous().view(batch_size * num_beams, -1)
        attention_mask = attention_mask.unsqueeze(1).expand(batch_size, num_beams, -1)
        attention_mask = attention_mask.contiguous().view(batch_size * num_beams, -1)
        
        # Initialize scores
        beam_scores = torch.zeros((batch_size, num_beams), device=device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)
        
        # Initialize cognitive states and expert tracking
        cognitive_states = torch.zeros(batch_size * num_beams, self.config.n_embd, device=device)
        expert_history = [{} for _ in range(batch_size * num_beams)]
        
        # Initialize dynamic cache for decoder
        past_key_values = None if not use_cache else [None] * len(self.h)
        
        # Keep track of which sequences are already finished
        done = [False for _ in range(batch_size)]
        
        with torch.no_grad():
            for step in range(max_length):
                # Model forward pass
                model_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "past_key_values": past_key_values,
                    "use_cache": use_cache,
                }
                outputs = self(**model_inputs)
                
                next_token_logits = outputs["logits"][:, -1, :] / temperature
                past_key_values = outputs.get("past_key_values", past_key_values)
                hidden_states = outputs.get("hidden_states", None)
                
                # Update cognitive states and expert confidence
                if hidden_states is not None:
                    for i, block in enumerate(self.h):
                        if hasattr(block, 'cognitive'):
                            new_cognitive_state = block.cognitive(hidden_states[:, -1:])[0].squeeze(1)
                            cognitive_influence = cognitive_state_similarity(new_cognitive_state, cognitive_states)
                            next_token_logits *= (1 + cognitive_influence * cognitive_memory_influence)
                            cognitive_states = cognitive_states * 0.9 + new_cognitive_state * 0.1
                        
                        if hasattr(block.mlp, 'gate'):
                            expert_logits = block.mlp.gate(hidden_states[:, -1])
                            for b in range(batch_size * num_beams):
                                expert_conf = get_expert_confidence(expert_logits[b], expert_history[b])
                                if expert_conf > expert_confidence_threshold:
                                    next_token_logits[b] *= (1 + expert_conf * 0.2)

                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(batch_size * num_beams):
                        for previous_token in set(input_ids[i].tolist()):
                            next_token_logits[i, previous_token] /= repetition_penalty
                
                if decoder_strategy == "beam_search":
                    next_token_logits = _beam_search_step(
                        next_token_logits, beam_scores, num_beams, 
                        num_beam_groups, diversity_penalty, batch_size
                    )
                elif decoder_strategy == "sample":
                    next_token_logits = _sample_step(next_token_logits, top_k, top_p, temperature)
                # For greedy, we use logits as is
                
                next_token_scores = F.log_softmax(next_token_logits, dim=-1)
                
                if decoder_strategy == "greedy":
                    next_tokens = torch.argmax(next_token_scores, dim=-1)
                    next_scores = next_token_scores.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)
                else:
                    next_scores, next_tokens = torch.topk(
                        next_token_scores, num_beams, dim=-1, 
                        largest=True, sorted=True
                    )
                
                # Next beam indices
                next_beam_indices = torch.div(next_tokens, vocab_size, rounding_mode='floor')
                next_tokens = next_tokens % vocab_size
                
                # Update beam indices
                beam_outputs = {
                    "next_beam_scores": next_scores.view(-1),
                    "next_beam_tokens": next_tokens.view(-1),
                    "next_beam_indices": next_beam_indices.view(-1),
                }
                
                # Prepare for next iteration
                input_ids = torch.cat([input_ids[beam_outputs["next_beam_indices"]], 
                                    beam_outputs["next_beam_tokens"].unsqueeze(-1)], dim=-1)
                
                attention_mask = torch.cat([
                    attention_mask[beam_outputs["next_beam_indices"]],
                    attention_mask.new_ones((attention_mask.shape[0], 1))
                ], dim=-1)
                
                beam_scores = beam_outputs["next_beam_scores"]
                
                # Update cognitive states and expert history for next iteration
                cognitive_states = cognitive_states[beam_outputs["next_beam_indices"]]
                expert_history = [expert_history[i] for i in beam_outputs["next_beam_indices"].tolist()]
                
                # Check if we're done
                for batch_idx in range(batch_size):
                    if not done[batch_idx]:
                        for beam_id in range(num_beams):
                            batch_beam_idx = batch_idx * num_beams + beam_id
                            if eos_token_id is not None and beam_outputs["next_beam_tokens"][batch_beam_idx].item() == eos_token_id:
                                if step >= min_length:
                                    generated_hyps[batch_idx].add(
                                        input_ids[batch_beam_idx].clone(),
                                        beam_scores[batch_beam_idx].item(),
                                        cognitive_states[batch_beam_idx].clone(),
                                        expert_history[batch_beam_idx].copy()
                                    )
                
                # Check if all beams are done
                done = [(hyp.num_beams == len(hyp.beams) and len(hyp.beams) >= num_return_sequences)
                    for hyp in generated_hyps]
                if all(done):
                    break
        
        # Select best hypotheses
        output_sequences = []
        output_cognitive_states = []
        output_expert_histories = []
        
        for i, hyp in enumerate(generated_hyps):
            sorted_hyps = sorted(hyp.beams, key=lambda x: x[0], reverse=True)
            for j in range(min(len(sorted_hyps), num_return_sequences)):
                output_sequences.append(sorted_hyps[j][1])
                if sorted_hyps[j][2] is not None:  # cognitive state
                    output_cognitive_states.append(sorted_hyps[j][2])
                if sorted_hyps[j][3] is not None:  # expert history
                    output_expert_histories.append(sorted_hyps[j][3])
        
        # Prepare output
        output_sequences = torch.stack(output_sequences) if len(output_sequences) > 1 else output_sequences[0].unsqueeze(0)
        
        return {
            "sequences": output_sequences,
            "cognitive_states": output_cognitive_states if output_cognitive_states else None,
            "expert_histories": output_expert_histories if output_expert_histories else None
        }

def _beam_search_step(logits, scores, num_beams, num_beam_groups, diversity_penalty, batch_size):
    vocab_size = logits.shape[-1]
    
    # Calculate log probabilities
    next_scores = F.log_softmax(logits, dim=-1)
    
    # Add diversity penalty between beam groups
    if num_beam_groups > 1:
        diversity_group_size = num_beams // num_beam_groups
        for batch_idx in range(batch_size):
            for group_idx in range(num_beam_groups):
                group_start = group_idx * diversity_group_size
                group_end = (group_idx + 1) * diversity_group_size
                group_scores = next_scores[batch_idx * num_beams + group_start:batch_idx * num_beams + group_end]
                
                # Apply diversity penalty
                for other_group_idx in range(num_beam_groups):
                    if other_group_idx != group_idx:
                        other_group_start = other_group_idx * diversity_group_size
                        other_group_end = (other_group_idx + 1) * diversity_group_size
                        other_group_scores = next_scores[batch_idx * num_beams + other_group_start:batch_idx * num_beams + other_group_end]
                        
                        penalty = torch.max(other_group_scores, dim=0)[0] * diversity_penalty
                        group_scores -= penalty
    
    return next_scores

def _sample_step(logits, top_k, top_p, temperature):
    # Top-k filtering
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
    
    # Top-p filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
    
    return logits

import os
import torch
from transformers import PreTrainedModel

def save_enhanced_model(model, optimizer, scheduler, tokenizer, config, epoch, train_loss_history, path):
    try:
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # 1. Save the model state
        model_to_save = model.module if hasattr(model, 'module') else model
        if isinstance(model_to_save, PreTrainedModel):
            model_to_save.save_pretrained(path, safe_serialization=False)
        else:
            torch.save(model_to_save.state_dict(), os.path.join(path, 'pytorch_model.bin'))
        
        # 2. Save the tokenizer
        if tokenizer is not None:
            tokenizer.save_pretrained(path)
        
        # 3. Save the config
        if config is not None:
            config.save_pretrained(path)
        
        # 4. Save the training state
        training_state = {
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_loss_history': train_loss_history,
        }
        
        # 5. Save additional components
        additional_components = {}
        for i, block in enumerate(model.h):
            if hasattr(block, 'mlp') and hasattr(block.mlp, 'experts'):
                additional_components.setdefault('moe_states', {})[i] = block.mlp.state_dict()
            if hasattr(block, 'ntk'):
                additional_components.setdefault('ntk_states', {})[i] = block.ntk.state_dict()
            if hasattr(block, 'decision_trees'):
                additional_components.setdefault('decision_tree_states', {})[i] = block.decision_trees.state_dict()
            if hasattr(block, 'cognitive'):
                additional_components.setdefault('cognitive_states', {})[i] = block.cognitive.state_dict()
        
        # Combine all states
        full_state = {
            'training_state': training_state,
            'additional_components': additional_components
        }
        
        # Save the combined state
        torch.save(full_state, os.path.join(path, 'training_state.bin'))
        
        print(f"Model and training state successfully saved to {path}")
    except Exception as e:
        print(f"Error occurred while saving the model: {str(e)}")
        raise

def load_enhanced_model(path, training=False):
    # 1. Load config and model
    config = DecoderOnlyConfig.from_pretrained(path)
    model = DecoderOnlyGPT.from_pretrained(path, config=config)
    
    # 2. Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(path)
    
    if not training:
        return model, tokenizer
    
    # 3. Load training state
    training_state_path = os.path.join(path, 'training_state.bin')
    if not os.path.exists(training_state_path):
        raise FileNotFoundError(f"Training state not found at {training_state_path}")
    
    full_state = torch.load(training_state_path)
    training_state = full_state['training_state']
    additional_components = full_state['additional_components']
    
    # 4. Restore additional components
    for i, block in enumerate(model.h):
        if hasattr(block.mlp, 'experts'):
            block.mlp.load_state_dict(additional_components['moe_states'][i])
        if hasattr(block, 'ntk'):
            block.ntk.load_state_dict(additional_components['ntk_states'][i])
        if hasattr(block, 'decision_trees'):
            block.decision_trees.load_state_dict(additional_components['decision_tree_states'][i])
        if hasattr(block, 'cognitive'):
            block.cognitive.load_state_dict(additional_components['cognitive_states'][i])
    
    # 5. Create and load optimizer
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    optimizer.load_state_dict(training_state['optimizer_state_dict'])
    
    # 6. Create and load scheduler if it exists
    scheduler = None
    if training_state['scheduler_state_dict'] is not None:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.num_training_steps
        )
        scheduler.load_state_dict(training_state['scheduler_state_dict'])
    
    # 7. Restore memory states if they exist
    if 'memory_states' in training_state:
        for i, block in enumerate(model.h):
            if hasattr(block, 'cognitive'):
                block.cognitive.memory_state = training_state['memory_states'][i]
    
    return model, tokenizer, optimizer, scheduler, training_state['epoch'], training_state['train_loss_history']

def load_enhanced_model(path, training=False):
    import os
    from transformers import GPT2Tokenizer
    
    # 1. Load config and model
    config = DecoderOnlyConfig.from_pretrained(path)
    model = DecoderOnlyGPT.from_pretrained(path, config=config)
    
    # 2. Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(path)
    
    if not training:
        return model, tokenizer
    
    # 3. Load training state
    training_state_path = os.path.join(path, 'training_state.bin')
    if not os.path.exists(training_state_path):
        raise FileNotFoundError(f"Training state not found at {training_state_path}")
    
    full_state = torch.load(training_state_path)
    training_state = full_state['training_state']
    additional_components = full_state['additional_components']
    
    # 4. Restore additional components
    for i, block in enumerate(model.h):
        if hasattr(block.mlp, 'experts'):
            block.mlp.load_state_dict(additional_components['moe_states'][i])
        if hasattr(block, 'ntk'):
            block.ntk.load_state_dict(additional_components['ntk_states'][i])
        if hasattr(block, 'decision_trees'):
            block.decision_trees.load_state_dict(additional_components['decision_tree_states'][i])
        if hasattr(block, 'cognitive'):
            block.cognitive.load_state_dict(additional_components['cognitive_states'][i])
    
    # 5. Create and load optimizer
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    optimizer.load_state_dict(training_state['optimizer_state_dict'])
    
    # 6. Create and load scheduler if it exists
    scheduler = None
    if training_state['scheduler_state_dict'] is not None:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.num_training_steps
        )
        scheduler.load_state_dict(training_state['scheduler_state_dict'])
    
    # 7. Restore memory states if they exist
    if 'memory_states' in training_state:
        for i, block in enumerate(model.h):
            if hasattr(block, 'cognitive'):
                block.cognitive.memory_state = training_state['memory_states'][i]
    
    # 8. Create and load GradScaler
    scaler = torch.amp.GradScaler()
    scaler.load_state_dict(training_state['scaler'])
    
    return model, tokenizer, optimizer, scheduler, training_state['epoch'], training_state['train_loss_history'], scaler

def train_chatbot_model(model, tokenizer, dataset, config, start_epoch=0, train_loss_history=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create DataLoader with gradient accumulation
    effective_batch_size = config.batch_size * config.gradient_accumulation_steps
    train_loader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        pin_memory=True
    )
    
    # Initialize optimizer with gradient clipping
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        eps=1e-8,  # Increased epsilon for numerical stability
        weight_decay=0.01  # L2 regularization to prevent extreme weights
    )
    
    num_training_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    warmup_steps = min(1000, num_training_steps // 10)  # Dynamic warmup
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    
    if train_loss_history is None:
        train_loss_history = []
    
    # Initialize gradient norm tracking
    grad_norm_moving_avg = 0.0
    beta = 0.98  # For exponential moving average
    
    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                outputs = model(
                    input_ids, 
                    attention_mask=attention_mask, 
                    labels=labels
                )
                loss = outputs['loss'] / config.gradient_accumulation_steps
                
                # Check for NaN loss before backward pass
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"NaN/Inf detected in loss at step {step}. Skipping batch.")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    # Calculate gradient norm
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    
                    # Update gradient norm moving average
                    grad_norm_moving_avg = beta * grad_norm_moving_avg + (1 - beta) * grad_norm
                    
                    # Adjust learning rate based on gradient norm
                    if grad_norm_moving_avg > config.max_grad_norm:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= 0.9
                    
                    # Step optimizer and scheduler
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                # Update progress bar
                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'loss': f"{loss.item() * config.gradient_accumulation_steps:.4f}",
                    'lr': f"{current_lr:.2e}",
                    'grad_norm': f"{grad_norm_moving_avg:.2f}",
                })
                
                total_loss += loss.item() * config.gradient_accumulation_steps
                
            except RuntimeError as e:
                print(f"Error during training: {str(e)}")
                continue
        
        avg_loss = total_loss / len(train_loader)
        train_loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{config.num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        save_enhanced_model(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            tokenizer=tokenizer,
            config=config,
            epoch=epoch+1,
            train_loss_history=train_loss_history,
            path=f"checkpoint_epoch_{epoch+1}"
        )
    
    return model, train_loss_history, scheduler, optimizer

class WikipediaDataset(Dataset):
    def __init__(self, tokenizer, max_length=512, num_examples=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load Wikipedia dataset
        dataset = load_dataset("wikipedia", "20220301.en", split="train[:2%]")
        
        # Use RecursiveCharacterTextSplitter from LangChain
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_length,
            chunk_overlap=18,
            length_function=len,
            separators=["\n\n", "\n", "."]
        )
        
        self.texts = []
        for item in tqdm(dataset, desc="Processing Wikipedia articles"):
            chunks = text_splitter.split_text(item['text'])
            self.texts.extend(chunks)
        
        # Limit the number of examples if specified
        if num_examples is not None:
            self.texts = self.texts[:num_examples]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize the text
        inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        
        # Create labels (shifted input_ids)
        labels = inputs['input_ids'].clone()
        labels[:, :-1] = inputs['input_ids'][:, 1:]
        labels[:, -1] = -100  # Ignore the last token prediction
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }

def main(num_examples=None, resume_from=None):
    # Training configuration
    training_config = {
        'batch_size': 2,
        'num_epochs': 3,
        'learning_rate': 1e-4,
        'warmup_steps': 100,
        'max_length': 512,
    }
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2",clean_up_tokenization_spaces=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize or load model and training state
    if resume_from:
        print(f"Resuming training from {resume_from}")
        model, tokenizer, optimizer, scheduler, start_epoch, train_loss_history = load_enhanced_model(
            resume_from, training=True
        )
    else:
        print("Initializing new model")
        config = DecoderOnlyConfig(
            vocab_size=tokenizer.vocab_size,
            n_positions=512,
            n_embd=512,
            n_layer=6,
            n_head=8,
            n_inner=1536,
            learning_rate=training_config['learning_rate'], 
            gradient_accumulation_steps=4,
            max_grad_norm=1.0,
            warmup_ratio=0.1,
            use_moe=True,
            num_experts=2,
            top_k_experts=2,
            use_cognitive_layer=True,
            use_ntk_layer=True,
            use_decision_trees=True,
            num_epochs=training_config['num_epochs'],
            batch_size=training_config['batch_size'],
            warmup_steps=training_config['warmup_steps'],
        )
        model = DecoderOnlyGPT(config)
        start_epoch = 0
        train_loss_history = None

    # Create dataset
    dataset = WikipediaDataset(
        tokenizer, 
        max_length=config.n_positions,
        num_examples=num_examples
    )

    # Train the model
    model, train_loss_history, scheduler, optimizer = train_chatbot_model(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        config=config,
        start_epoch=start_epoch,
        train_loss_history=train_loss_history
    )

    # Save the final model
    save_enhanced_model(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            tokenizer=tokenizer,
            config=config,
            epoch=config.num_epochs,
            train_loss_history=train_loss_history,
            path="final_enhanced_wikipedia_chatbot"
    )

    return trained_model, tokenizer, train_loss_history

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train enhanced chatbot model')
    parser.add_argument('--num_examples', type=int, default=None, 
                        help='Number of examples to use for training (default: all)')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    args = parser.parse_args()

    trained_model, tokenizer, loss_history = main(
        num_examples=args.num_examples,
        resume_from=args.resume_from
    )