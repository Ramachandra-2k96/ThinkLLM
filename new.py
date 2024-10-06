import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer
from datasets import load_dataset
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm
import os
import json
import math
from torch.nn import functional as F

class EnhancedCustomConfig(PretrainedConfig):
    model_type = "enhanced_custom"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=1024,
        num_hidden_layers=16,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=1024,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        num_tree_layers=4,
        num_thought_steps=3,
        tie_word_embeddings=True,
        use_rezero=True,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs)

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

class ReZeroTransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, dropout=config.attention_probs_dropout_prob, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.alpha = nn.Parameter(torch.tensor(1e-3))
        
    def forward(self, x, attention_mask=None):
        attention_output, _ = self.attention(x, x, x, key_padding_mask=attention_mask)
        x = x + self.alpha * attention_output
        x = x + self.alpha * self.feed_forward(x)
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

class EnhancedTreeNode(nn.Module):
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

class EnhancedTreeLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.nodes = nn.ModuleList([EnhancedTreeNode(config) for _ in range(2**config.num_tree_layers - 1)])
    
    def forward(self, x):
        for node in self.nodes:
            x = node(x)
        return x

class EnhancedCustomModel(PreTrainedModel):
    config_class = EnhancedCustomConfig
    base_model_prefix = "enhanced_custom"

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.layers = nn.ModuleList([
            nn.ModuleList([
                ReZeroTransformerLayer(config) if config.use_rezero else nn.TransformerEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.num_attention_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=config.hidden_dropout_prob,
                    activation=config.hidden_act,
                    batch_first=True
                ),
                EnhancedTreeLayer(config),
                ThoughtLayer(config)
            ])
            for _ in range(config.num_hidden_layers)
        ])
        
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self._tie_or_clone_weights(self.lm_head, self.word_embeddings)

        self.post_init()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, value):
        self.word_embeddings = value

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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.embedding_dropout(embeddings)

        hidden_states = embeddings
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states = layer[0](hidden_states, attention_mask=(attention_mask == 0))
            hidden_states = layer[1](hidden_states)
            hidden_states = layer[2](hidden_states)

            if output_attentions:
                all_attentions = all_attentions + (hidden_states,)

        hidden_states = self.final_layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + (hidden_states,)
            return ((loss,) + output) if loss is not None else output

        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': all_hidden_states,
            'attentions': all_attentions
        }

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

import gc
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

def save_enhanced_custom_model(model, tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    model.config.save_pretrained(output_dir)

    state_dict = model.state_dict()
    
    if 'lm_head.weight' in state_dict and torch.equal(state_dict['lm_head.weight'], state_dict['word_embeddings.weight']):
        del state_dict['lm_head.weight']
    
    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))

    tokenizer.save_pretrained(output_dir)

    custom_components = {
        "tree_layers": [layer[1].state_dict() for layer in model.layers],
        "thought_layers": [layer[2].state_dict() for layer in model.layers]
    }
    torch.save(custom_components, os.path.join(output_dir, "custom_components.pt"))

    with open(os.path.join(output_dir, "tied_weights_info.json"), "w") as f:
        json.dump({"lm_head_tied_to_embeddings": True}, f)

    print(f"Model saved to {output_dir}")

def evaluate(model, eval_loader, device):
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

def train_enhanced_custom_model(model, dataset, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=len(train_loader) * config.epochs)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]")
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(train_pbar):
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)

            with autocast('cuda', dtype=torch.float16):
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
            train_pbar.set_postfix({'loss': loss.item() * config.accumulation_steps, 'lr': scheduler.get_last_lr()[0]})

            # Move this block inside the loop to free up memory more frequently
            del input_ids, attention_mask, outputs, loss
            torch.cuda.empty_cache()

        print(f"Epoch {epoch+1}/{config.epochs}, Train Loss: {total_loss/len(train_loader):.4f}")

        # Add explicit garbage collection after each epoch
        gc.collect()
        torch.cuda.empty_cache()

def main():
    config = EnhancedCustomConfig(
        vocab_size=30522,
        hidden_size=256,  # Reduced from 1024
        num_hidden_layers=8,  # Reduced from 16
        num_attention_heads=8,  # Reduced from 16
        num_tree_layers=4,  # Reduced from 4
        num_thought_steps=4,  # Reduced from 3
        max_position_embeddings=512,  # Reduced from 1024
        tie_word_embeddings=True,
        use_rezero=True
    )

    # Add training-specific configurations
    config.batch_size = 20
    config.epochs = 2
    config.learning_rate = 1e-4
    config.accumulation_steps = 4000
    config.warmup_steps = 200

    model = EnhancedCustomModel(config)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", clean_up_tokenization_spaces=True)

    # Reduce subset size for faster processing
    dataset = WikipediaDataset(tokenizer, max_length=128, subset_size=50000)

    train_enhanced_custom_model(model, dataset, config)

    # Save the trained model
    save_enhanced_custom_model(model, tokenizer, "enhanced_custom_model_final")

if __name__ == "__main__":
    main()