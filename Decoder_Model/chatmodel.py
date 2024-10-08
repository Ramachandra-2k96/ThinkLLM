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
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

# Update the DecoderOnlyConfig class
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

# Update the save and load functionality
def save_enhanced_model(model, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.config,
    }, path)

def load_enhanced_model(path):
    checkpoint = torch.load(path)
    config = checkpoint['config']
    model = DecoderOnlyGPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

# Update the training function
def train_chatbot_model(model, dataset, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = model.to(device)
    except RuntimeError:
        print("Not enough GPU memory. Falling back to CPU.")
        device = torch.device("cpu")
        model = model.to(device)

    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    
    num_training_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps
    )

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        memory_states = [None] * len(model.h)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
            loss = outputs['loss']
            memory_states = outputs.get('memory_states', [None] * len(model.h))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / (progress_bar.n + 1):.4f}",
                'lr': f"{current_lr:.2e}"
            })
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{config.num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint after each epoch
        save_enhanced_model(model, f"checkpoint_epoch_{epoch+1}.pt")
    
    return model


class WikipediaDataset(Dataset):
    def __init__(self, tokenizer, max_length=512, num_examples=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load Wikipedia dataset
        dataset = load_dataset("wikipedia", "20220301.en", split="train[:1%]")
        
        # Use RecursiveCharacterTextSplitter from LangChain
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
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

def main(num_examples=None):
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2",clean_up_tokenization_spaces=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize enhanced model configuration
    config = DecoderOnlyConfig(
        vocab_size=tokenizer.vocab_size,
        n_positions=512,
        n_embd=768,
        n_layer=4,       # Reduced from 8
        n_head=8,
        n_inner=3072,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_moe=True,
        num_experts=2,   # Reduced from 4
        top_k_experts=2,
        use_cognitive_layer=True,
        use_ntk_layer=True,
        use_decision_trees=True
    )
    
    # Create enhanced model
    model = DecoderOnlyGPT(config)
    
    # Create dataset
    dataset = WikipediaDataset(tokenizer, num_examples=num_examples)
    
    # Training configuration
    train_config = type('TrainConfig', (), {
        'batch_size': 1,
        'learning_rate': 5e-5,
        'num_epochs': 3,
        'warmup_steps': 100
    })()
    
    # Train the model
    print("Training enhanced model...")
    trained_model = train_chatbot_model(model, dataset, train_config)
    
    # Save the final trained model
    save_enhanced_model(trained_model, "enhanced_wikipedia_chatbot_model.pt")
    tokenizer.save_pretrained("enhanced_wikipedia_chatbot_tokenizer")
    
    print("\nTraining completed. Starting chat interface...")

if __name__ == "__main__":
    main(num_examples=100)  # Set to None to use the entire dataset