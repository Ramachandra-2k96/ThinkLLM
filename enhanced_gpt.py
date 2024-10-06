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
#### 2'ND ATTEMPT
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
        return_dict=None,
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

def save_enhanced_custom_model_v2(model, tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    model.config.save_pretrained(output_dir)

    state_dict = model.state_dict()
    
    if 'lm_head.weight' in state_dict and 'embeddings.word_embeddings.weight' in state_dict:
        if torch.equal(state_dict['lm_head.weight'], state_dict['embeddings.word_embeddings.weight']):
            del state_dict['lm_head.weight']
    
    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))

    tokenizer.save_pretrained(output_dir)

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

    train_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True,
        drop_last=True  # Prevent issues with last batch having different size
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

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]")
        
        for i, batch in enumerate(train_pbar):
            try:
                input_ids = batch['input_ids'].squeeze(1).to(device)
                attention_mask = batch['attention_mask'].squeeze(1).to(device)
                
                # Ensure input dimensions are correct
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                if attention_mask.dim() == 1:
                    attention_mask = attention_mask.unsqueeze(0)
                
                with autocast(device_type=device.type, dtype=torch.float16):
                    outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
                    loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
                    loss = loss / config.accumulation_steps

                scaler.scale(loss).backward()

                if (i + 1) % config.accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                total_loss += loss.item() * config.accumulation_steps
                train_pbar.set_postfix({
                    'loss': loss.item() * config.accumulation_steps,
                    'lr': scheduler.get_last_lr()[0]
                })

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"CUDA out of memory in batch {i}. Skipping batch and clearing cache.")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    optimizer.zero_grad(set_to_none=True)
                    continue
                else:
                    print(f"Error in batch {i}: {str(e)}")
                    traceback.print_exc()
                    continue

            # Clean up to prevent memory leaks
            del input_ids, attention_mask, outputs, loss
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{config.epochs}, Train Loss: {avg_loss:.4f}")

        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

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
        return self.tokenizer(self.data[idx], max_length=self.max_length, truncation=True, padding='max_length', return_tensors="pt")
 

def main():
    config = EnhancedCustomConfigV2(
        vocab_size=30522,
        hidden_size=256,  # Must be divisible by num_attention_heads
        num_hidden_layers=8,  # Reduced for memory
        num_attention_heads=8,  # 256 / 8 = 32, which is good
        intermediate_size=1024,
        num_tree_layers=2,
        num_thought_steps=2,
        max_position_embeddings=512,
        tie_word_embeddings=True,
        use_rezero=True,
        use_adapter=True,
        adapter_size=32,
        use_gated_ffn=True,
        use_moe=True,  # Disabled for initial testing
        num_experts=4,
        top_k_experts=4,
        use_sparse_attention=True,
        sparse_attention_window=256,
        use_dynamic_ntk=True,  # Disabled for initial testing
        ntk_alpha=0.5,
        use_mixture_of_experts=True,  # Disabled for initial testing
        num_moe_experts=4,
        moe_top_k=4
    )

    # Update training configuration
    config.batch_size = 4
    config.epochs = 1
    config.learning_rate = 1e-4  # Reduced learning rate
    config.accumulation_steps = 4
    config.warmup_steps = 100

    model = EnhancedCustomModelV2(config)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased",clean_up_tokenization_spaces=True)

    # Start with a very small dataset for testing
    dataset = WikipediaDataset(tokenizer, max_length=512, subset_size=1000)

    try:
        model = train_enhanced_custom_model_v2(model, dataset, config)
        save_enhanced_custom_model_v2(model, tokenizer, "enhanced_custom_model_v2_final")
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()