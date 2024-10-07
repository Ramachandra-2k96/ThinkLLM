import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from transformers import (
    PreTrainedModel, 
    PretrainedConfig, 
    AutoTokenizer, 
    get_cosine_schedule_with_warmup,
    LogitsProcessor, 
    LogitsProcessorList, 
    StoppingCriteria, 
    StoppingCriteriaList
)
from datasets import load_dataset
from tqdm.auto import tqdm
from typing import Dict, List, Optional, Union
import os
import math

class EnhancedModelConfig(PretrainedConfig):
    model_type = "enhanced_encoder_decoder"
    
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_layers=6,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        dropout_prob=0.1,
        max_position_embeddings=512,
        num_tree_layers=4,
        adapter_size=64,
        num_experts=4,
        sparse_window=256,
        ntk_alpha=0.5,
        batch_size=4,
        learning_rate=5e-5,
        num_thought_paths=4,
        tree_depth=3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.dropout_prob = dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.num_tree_layers = num_tree_layers
        self.adapter_size = adapter_size
        self.num_experts = num_experts
        self.sparse_window = sparse_window
        self.ntk_alpha = ntk_alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_thought_paths = num_thought_paths
        self.tree_depth = tree_depth

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.sparse_window = config.sparse_window

    def forward(self, x, mask=None):
        B, L, D = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2), qkv)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if self.sparse_window < L:
            sparse_mask = torch.ones(L, L, dtype=torch.bool, device=x.device).triu(-self.sparse_window).tril(self.sparse_window)
            scores.masked_fill_(~sparse_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        if mask is not None:
            scores += mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)

class MixtureOfExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size),
                nn.GELU(),
                nn.Linear(config.intermediate_size, config.hidden_size)
            ) for _ in range(config.num_experts)
        ])
        self.gate = nn.Linear(config.hidden_size, config.num_experts)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, x):
        # x shape: [batch_size, sequence_length, hidden_size]
        batch_size, sequence_length, hidden_size = x.shape
        
        # Compute gate weights
        gate_logits = self.gate(x)  # shape: [batch_size, sequence_length, num_experts]
        expert_weights = F.softmax(gate_logits, dim=-1)
        
        # Apply experts
        expert_outputs = torch.stack([expert(x) for expert in self.experts])
        # expert_outputs shape: [num_experts, batch_size, sequence_length, hidden_size]
        
        # Combine expert outputs
        output = torch.einsum('bse,ebsd->bsd', expert_weights, expert_outputs)
        return self.dropout(output)

class Adapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.down = nn.Linear(config.hidden_size, config.adapter_size)
        self.up = nn.Linear(config.adapter_size, config.hidden_size)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))

class ThoughtNode(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decision = nn.Linear(config.hidden_size, config.num_thought_paths)
        self.thought_paths = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size)
            for _ in range(config.num_thought_paths)
        ])
        self.combine = nn.Linear(config.hidden_size * config.num_thought_paths, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, x):
        decisions = F.softmax(self.decision(x), dim=-1)  # shape: [batch_size, seq_len, num_thought_paths]
        thoughts = torch.stack([path(x) for path in self.thought_paths], dim=-2)  # shape: [batch_size, seq_len, num_thought_paths, hidden_size]
        
        # Reshape decisions to match thoughts for broadcasting
        decisions = decisions.unsqueeze(-1)  # shape: [batch_size, seq_len, num_thought_paths, 1]
        
        weighted_thoughts = decisions * thoughts  # shape: [batch_size, seq_len, num_thought_paths, hidden_size]
        combined = weighted_thoughts.flatten(start_dim=-2)  # shape: [batch_size, seq_len, num_thought_paths * hidden_size]
        
        return self.dropout(self.combine(combined))

class DecisionTreeLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.depth = config.tree_depth
        self.hidden_size = config.hidden_size
        
        self.decision_nodes = nn.ModuleList([
            nn.Linear(config.hidden_size, 2) for _ in range(2 ** self.depth - 1)
        ])
        
        self.leaf_nodes = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size) for _ in range(2 ** self.depth)
        ])

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        decisions = []
        for level in range(self.depth):
            level_decisions = []
            start_idx = 2 ** level - 1
            num_nodes = 2 ** level
            
            for node in range(num_nodes):
                node_idx = start_idx + node
                decision = torch.sigmoid(self.decision_nodes[node_idx](x))
                level_decisions.append(decision)
            
            level_decisions = torch.stack(level_decisions, dim=2)
            decisions.append(level_decisions)
        
        path_probs = torch.ones((batch_size, seq_len, 1), device=x.device)
        for level_decisions in decisions:
            left_probs = level_decisions[:, :, :, 0]
            right_probs = level_decisions[:, :, :, 1]
            path_probs = torch.cat([path_probs * left_probs, path_probs * right_probs], dim=-1)
        
        leaf_outputs = torch.stack([leaf(x) for leaf in self.leaf_nodes], dim=2)
        
        output = torch.sum(leaf_outputs * path_probs.unsqueeze(-1), dim=2)
        return output

class DynamicNTKLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ntk_alpha = config.ntk_alpha
        self.feature_map = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads)

    def forward(self, x):
        phi_x = self.feature_map(x)
        kernel = torch.einsum('bid,bjd->bij', phi_x, phi_x)
        
        scaled_kernel = kernel * self.ntk_alpha
        
        attended_output, _ = self.attention(x, x, x)
        
        return x + scaled_kernel @ attended_output

class EnhancedLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.moe = MixtureOfExperts(config)
        self.adapter = Adapter(config)
        self.thought_node = ThoughtNode(config)
        self.decision_tree = DecisionTreeLayer(config)
        self.dynamic_ntk = DynamicNTKLayer(config)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.norm3 = nn.LayerNorm(config.hidden_size)
        self.norm4 = nn.LayerNorm(config.hidden_size)
        self.norm5 = nn.LayerNorm(config.hidden_size)

    def forward(self, x, mask=None):
        x = x + self.adapter(self.attention(self.norm1(x), mask))
        x = x + self.moe(self.norm2(x))
        x = x + self.thought_node(self.norm3(x))
        x = x + self.decision_tree(self.norm4(x))
        x = x + self.dynamic_ntk(self.norm5(x))
        return x

class EnhancedModel(PreTrainedModel):
    config_class = EnhancedModelConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        self.embeddings = nn.ModuleDict({
            'word': nn.Embedding(config.vocab_size, config.hidden_size),
            'position': nn.Embedding(config.max_position_embeddings, config.hidden_size)
        })
        
        self.encoder = nn.ModuleList([EnhancedLayer(config) for _ in range(config.num_layers)])
        self.decoder = nn.ModuleList([EnhancedLayer(config) for _ in range(config.num_layers)])
        
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embeddings['word'].weight
        
        self.dropout = nn.Dropout(config.dropout_prob)
        self.init_weights()

    def get_embeddings(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        
        # Ensure position_ids are within the allowed range
        position_ids = position_ids.clamp(0, self.config.max_position_embeddings - 1)
        
        word_embeds = self.embeddings['word'](input_ids)
        pos_embeds = self.embeddings['position'](position_ids)
        
        return self.dropout(word_embeds + pos_embeds)

    def encode(self, input_ids, attention_mask=None):
        x = self.get_embeddings(input_ids)
        
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.unsqueeze(1).unsqueeze(2)) * torch.finfo(x.dtype).min
        
        for layer in self.encoder:
            x = layer(x, attention_mask)
        return x

    def decode(self, input_ids, encoder_outputs, attention_mask=None):
        x = self.get_embeddings(input_ids)
        
        for layer in self.decoder:
            x = layer(x, attention_mask)
        
        return x

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        labels=None,
        **kwargs
    ):
        encoder_outputs = self.encode(input_ids, attention_mask)
        
        if decoder_input_ids is None:
            decoder_input_ids = input_ids
        
        decoder_outputs = self.decode(decoder_input_ids, encoder_outputs, attention_mask)
        logits = self.lm_head(decoder_outputs)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        
        return {'loss': loss, 'logits': logits} if loss is not None else {'logits': logits}

class WikipediaDataset(Dataset):
    def __init__(self, tokenizer, chunk_size=512, subset_size=None):
        self.tokenizer = tokenizer
        dataset = load_dataset("wikipedia", "20220301.en", split="train")
        if subset_size:
            dataset = dataset.select(range(min(subset_size, len(dataset))))
        
        self.chunks = []
        for text in tqdm(dataset["text"], desc="Processing dataset"):
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            self.chunks.extend(chunks)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        encodings = self.tokenizer(
            self.chunks[idx],
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in encodings.items()}

def train_model(model, dataset, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    train_size = int(0.9 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    num_training_steps = len(train_loader) * config.num_train_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_training_steps//10, num_training_steps=num_training_steps)
    
    scaler = GradScaler()
    best_val_loss = float('inf')
    
    for epoch in range(config.num_train_epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            with autocast(device.type, dtype=torch.float16):
                outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs['loss']
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
        
        val_loss = validate_model(model, val_loader, device)
        print(f"Epoch {epoch+1}: Train loss = {total_loss/len(train_loader):.4f}, Val loss = {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, "best_model")

def validate_model(model, val_loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            total_loss += outputs['loss'].item()
    
    return total_loss / len(val_loader)

def save_model(model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

def main():
    tokenizer = AutoTokenizer.from_pretrained("gpt2",clean_up_tokenization_spaces=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    config = EnhancedModelConfig(
    vocab_size=tokenizer.vocab_size,
    intermediate_size=1024,
    hidden_size=384,#768
    num_layers=6,
    num_attention_heads=12,#12
    batch_size=1,
    num_train_epochs=1,
    num_thought_paths=4,
    tree_depth=3,
)
    
    model = EnhancedModel(config)
    dataset = WikipediaDataset(tokenizer, subset_size=200)
    
    train_model(model, dataset, config)

if __name__ == "__main__":
    main()