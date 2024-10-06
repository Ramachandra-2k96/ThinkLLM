import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm
import os
import json
import math
from torch.nn import functional as F
import gc
import traceback
from typing import List, Optional, Union
from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteriaList, StoppingCriteria
#### 2'ND ATTEMPT
class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, penalty: float):
        self.penalty = penalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        score = torch.gather(scores, 1, input_ids)
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)
        scores.scatter_(1, input_ids, score)
        return scores
    
class MaxLengthCriteria(StoppingCriteria):
    def __init__(self, max_length: int):
        self.max_length = max_length

    def __call__(self, input_ids: torch.LongTensor, scores: Optional[torch.FloatTensor] = None) -> bool:
        return input_ids.shape[-1] >= self.max_length
    

class EnhancedCustomConfigV2(PretrainedConfig):
    model_type = "enhanced_custom_v2"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        num_tree_layers=4,
        num_thought_steps=4,
        tie_word_embeddings=True,
        use_rezero=True,
        use_adapter=True,
        adapter_size=64,
        use_gated_ffn=True,
        use_moe=True,
        num_experts=4,
        top_k_experts=2,
        use_sparse_attention=True,
        sparse_attention_window=256,
        use_dynamic_ntk=True,
        ntk_alpha=0.5,
        use_mixture_of_experts=True,
        num_moe_experts=8,
        moe_top_k=2,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.num_tree_layers = num_tree_layers
        self.num_thought_steps = num_thought_steps
        self.use_rezero = use_rezero
        self.use_adapter = use_adapter
        self.adapter_size = adapter_size
        self.use_gated_ffn = use_gated_ffn
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.use_sparse_attention = use_sparse_attention
        self.sparse_attention_window = sparse_attention_window
        self.use_dynamic_ntk = use_dynamic_ntk
        self.ntk_alpha = ntk_alpha
        self.use_mixture_of_experts = use_mixture_of_experts
        self.num_moe_experts = num_moe_experts
        self.moe_top_k = moe_top_k

class Adapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.down_project = nn.Linear(config.hidden_size, config.adapter_size)
        self.up_project = nn.Linear(config.adapter_size, config.hidden_size)
        self.activation = nn.GELU()

    def forward(self, hidden_states):
        down_projected = self.down_project(hidden_states)
        activated = self.activation(down_projected)
        up_projected = self.up_project(activated)
        return hidden_states + up_projected

class GatedFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        intermediate_output = self.intermediate(hidden_states)
        gate_output = self.gate(hidden_states)
        gated_output = self.activation(intermediate_output) * torch.sigmoid(gate_output)
        output = self.output(gated_output)
        return self.layer_norm(hidden_states + self.dropout(output))

class EnhancedAttentionV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.use_sparse_attention = config.use_sparse_attention
        self.sparse_attention_window = config.sparse_attention_window

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.size()

        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Expand attention_mask to match the shape of attention_scores
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.expand(-1, self.num_attention_heads, -1, -1)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            attention_scores = attention_scores + extended_attention_mask

        if self.use_sparse_attention:
            sparse_mask = torch.ones(seq_length, seq_length, dtype=torch.bool, device=hidden_states.device)
            for i in range(seq_length):
                start = max(0, i - self.sparse_attention_window // 2)
                end = min(seq_length, i + self.sparse_attention_window // 2)
                sparse_mask[i, start:end] = False
            sparse_mask = sparse_mask.unsqueeze(0).unsqueeze(1).expand(batch_size, self.num_attention_heads, -1, -1)
            attention_scores.masked_fill_(sparse_mask, -10000.0)

        # Normalize the attention scores to probabilities
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return self.layer_norm(hidden_states + context_layer)

class DecisionTreeNode(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decision = nn.Linear(config.hidden_size, 1)
        self.left = nn.Linear(config.hidden_size, config.hidden_size)
        self.right = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(self, x):
        decision = torch.sigmoid(self.decision(x))
        left_output = self.activation(self.left(x))
        right_output = self.activation(self.right(x))
        output = decision * left_output + (1 - decision) * right_output
        return self.layer_norm(output + x)

class DecisionTreeLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.nodes = nn.ModuleList([DecisionTreeNode(config) for _ in range(2**config.num_tree_layers - 1)])
    
    def forward(self, x):
        for node in self.nodes:
            x = node(x)
        return x

class ThoughtLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.thought_steps = config.num_thought_steps
        self.transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        
    def forward(self, x):
        original_x = x
        for _ in range(self.thought_steps):
            x = self.activation(self.transform(x))
        return x + original_x

import torch
import torch.nn as nn
import torch.nn.functional as F

class MixtureOfExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_moe_experts
        self.top_k = config.moe_top_k
        self.hidden_size = config.hidden_size
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size)
        ) for _ in range(self.num_experts)])
        self.gate = nn.Linear(config.hidden_size, self.num_experts)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x):
        # x shape: [batch_size, seq_len, hidden_size]
        batch_size, seq_len, _ = x.shape
        
        # Compute expert weights
        expert_weights = F.softmax(self.gate(x), dim=-1)
        # expert_weights shape: [batch_size, seq_len, num_experts]
        
        # Get top-k expert weights and indices
        top_k_weights, top_k_indices = torch.topk(expert_weights, self.top_k, dim=-1)
        # top_k_weights shape: [batch_size, seq_len, top_k]
        # top_k_indices shape: [batch_size, seq_len, top_k]
        
        # Normalize top-k weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # Combine expert outputs
        combined_output = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, :, k]
            expert_weight = top_k_weights[:, :, k].unsqueeze(-1)
            
            for i in range(self.num_experts):
                # Create a mask for the current expert
                expert_mask = (expert_idx == i)
                if expert_mask.any():
                    # Select inputs for this expert
                    expert_input = x[expert_mask]
                    # Apply the expert
                    expert_output = self.experts[i](expert_input)
                    # Add the weighted output to the result
                    combined_output[expert_mask] += expert_weight[expert_mask] * expert_output
        
        # Apply dropout and layer normalization
        output = self.dropout(combined_output)
        return self.layer_norm(x + output)
    
class DynamicNTKLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.alpha = config.ntk_alpha
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.GELU()

    def forward(self, x):
        h = self.linear(x)
        h_norm = h.norm(dim=-1, keepdim=True)
        x_norm = x.norm(dim=-1, keepdim=True)
        cos_theta = (h * x).sum(dim=-1, keepdim=True) / (h_norm * x_norm)
        ntk = (1 - cos_theta.pow(2)).sqrt() * h_norm / x_norm
        return x + self.alpha * self.activation(ntk * h)

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

class EnhancedCustomModelV2(PreTrainedModel):
    config_class = EnhancedCustomConfigV2
    base_model_prefix = "enhanced_custom_v2"

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = nn.ModuleDict({
            'word_embeddings': nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id),
            'position_embeddings': nn.Embedding(config.max_position_embeddings, config.hidden_size),
            'token_type_embeddings': nn.Embedding(config.type_vocab_size, config.hidden_size),
        })
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.encoder = nn.ModuleList([EnhancedLayerV2(config) for _ in range(config.num_hidden_layers)])
        
        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh()
        )
        
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self._tie_or_clone_weights(self.lm_head, self.embeddings['word_embeddings'])

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings['word_embeddings']

    def set_input_embeddings(self, value):
        self.embeddings['word_embeddings'] = value

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        output_embeddings.weight = input_embeddings.weight

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if position_ids is None:
            position_ids = torch.arange(0, input_shape[1], dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if inputs_embeds is None:
            inputs_embeds = self.embeddings['word_embeddings'](input_ids)
        position_embeddings = self.embeddings['position_embeddings'](position_ids)
        token_type_embeddings = self.embeddings['token_type_embeddings'](token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        hidden_states = self.embedding_dropout(embeddings)

        for layer in self.encoder:
            hidden_states = layer(hidden_states, attention_mask)

        pooled_output = self.pooler(hidden_states[:, 0])
        
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits, pooled_output, hidden_states)
            return ((loss,) + output) if loss is not None else output

        return {
            'loss': loss,
            'logits': logits,
            'pooled_output': pooled_output,
            'hidden_states': hidden_states,
        }
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_length: int = 50,
        min_length: int = 10,
        do_sample: bool = True,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        length_penalty: float = 1.0,
        pad_token_id: int = None,
        eos_token_id: int = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> torch.LongTensor:
        """
        Generate text based on input_ids.
        """
        # Set default token IDs if not provided
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        
        # Initialize variables
        batch_size = input_ids.shape[0]
        cur_len = input_ids.shape[1]
        
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        
        while cur_len < max_length:
            # Prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, attention_mask=attention_mask)
            
            # Forward pass
            outputs = self(**model_inputs)
            next_token_logits = outputs['logits'][:, -1, :]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in set(input_ids[i].tolist()):
                        next_token_logits[i, previous_token] /= repetition_penalty
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                for batch_idx in range(batch_size):
                    indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                    next_token_logits[batch_idx, indices_to_remove] = float('-inf')
            
            # Sample next token
            if do_sample:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # Finished sentences should have their next token be a padding token
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            
            # Update input_ids and attention_mask
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((batch_size, 1))], dim=-1)
            
            # Update unfinished sequences
            unfinished_sequences = unfinished_sequences.mul(next_tokens.ne(eos_token_id).long())
            
            # Stop when there are no sequences to generate
            if unfinished_sequences.max() == 0:
                break
            
            cur_len = cur_len + 1
        
        return input_ids

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

def evaluate_v2(model, eval_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
            total_loss += loss.item()
    
    return total_loss / len(eval_loader)

def train_enhanced_custom_model_v2(model, dataset, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create a validation dataset
    dataset_size = len(dataset)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    num_training_steps = len(train_loader) * config.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    scaler = GradScaler(enabled=(device.type == "cuda"))
    
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]")
        
        for i, batch in enumerate(train_pbar):
            optimizer.zero_grad(set_to_none=True)
            
            try:
                input_ids = batch['input_ids'].squeeze(1).to(device)
                attention_mask = batch['attention_mask'].squeeze(1).to(device)
                
                # Create shifted labels for next token prediction
                labels = input_ids.clone()
                labels = torch.roll(labels, shifts=-1, dims=1)
                labels[:, -1] = -100  # Ignore last token prediction
                
                with autocast(device_type=device.type, dtype=torch.float16):
                    outputs = model(
                        input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
                
                scaler.scale(loss).backward()
                
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                total_loss += loss.item()
                train_pbar.set_postfix({
                    'loss': loss.item(),
                    'lr': scheduler.get_last_lr()[0]
                })

            except RuntimeError as e:
                print(f"Error in batch {i}: {str(e)}")
                continue

        avg_train_loss = total_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].squeeze(1).to(device)
                attention_mask = batch['attention_mask'].squeeze(1).to(device)
                labels = input_ids.clone()
                labels = torch.roll(labels, shifts=-1, dims=1)
                labels[:, -1] = -100
                
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{config.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            save_enhanced_custom_model_v2(model, None, f"enhanced_custom_model_v2_best")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    return model

class WikipediaDataset(Dataset):
    def __init__(self, tokenizer, max_length=128, subset_size=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        dataset = load_dataset("wikipedia", "20220301.en", split="train")
        if subset_size is not None:
            dataset = dataset.select(range(min(subset_size, len(dataset))))
        
        self.data = dataset["text"]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # GPT-2 tokenization process
        return self.tokenizer(
            self.data[idx],
            max_length=self.max_length,
            truncation=True,
            padding='max_length',  
            return_tensors="pt"
        )
 
def save_enhanced_custom_model_v2(model, tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    model.config.save_pretrained(output_dir)

    state_dict = model.state_dict()
    
    if 'lm_head.weight' in state_dict and 'embeddings.word_embeddings.weight' in state_dict:
        if torch.equal(state_dict['lm_head.weight'], state_dict['embeddings.word_embeddings.weight']):
            del state_dict['lm_head.weight']
    
    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))

    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    else:
        print("Tokenizer not provided. Skipping tokenizer saving.")

    custom_components = {
        "tree_layers": [layer.tree.state_dict() for layer in model.encoder],
        "thought_layers": [layer.thought.state_dict() for layer in model.encoder],
        "moe_layers": [layer.moe.state_dict() for layer in model.encoder if hasattr(layer, 'moe')],
        "dynamic_ntk_layers": [layer.dynamic_ntk.state_dict() for layer in model.encoder if hasattr(layer, 'dynamic_ntk')]
    }
    torch.save(custom_components, os.path.join(output_dir, "custom_components.pt"))

    with open(os.path.join(output_dir, "tied_weights_info.json"), "w") as f:
        json.dump({"lm_head_tied_to_embeddings": True}, f)

    print(f"Model saved to {output_dir}")

def main():
    tokenizer = AutoTokenizer.from_pretrained("gpt2", clean_up_tokenization_spaces=True)
    tokenizer.pad_token = tokenizer.eos_token
    config = EnhancedCustomConfigV2(
        vocab_size = tokenizer.vocab_size,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=1024,
        num_tree_layers=2,
        num_thought_steps=2,
        max_position_embeddings=512,
        tie_word_embeddings=True,
        use_rezero=True,
        use_adapter=True,
        adapter_size=32,
        use_gated_ffn=True,
        use_moe=True,
        num_experts=2,
        top_k_experts=2,
        use_sparse_attention=True,
        sparse_attention_window=256,
        use_dynamic_ntk=True,
        ntk_alpha=0.5,
        use_mixture_of_experts=True,
        num_moe_experts=2,
        moe_top_k=2
    )

    # Update training configuration
    config.batch_size = 3
    config.epochs = 8
    config.learning_rate = 3e-5
    config.accumulation_steps = 4
    config.warmup_steps = 100

    model = EnhancedCustomModelV2(config)
    # Start with a very small dataset for testing
    dataset = WikipediaDataset(tokenizer, max_length=512, subset_size=10000)

    try:
        model = train_enhanced_custom_model_v2(model, dataset, config)
        save_enhanced_custom_model_v2(model, tokenizer, "enhanced_custom_model_v2_final")
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()