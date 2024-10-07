import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig, GPT2Tokenizer
from torch.amp import autocast, GradScaler
from datasets import load_dataset
from torch.utils.data import DataLoader
import math
from tqdm import tqdm
import wandb
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter


class CustomLLMConfig(PretrainedConfig):
    model_type = "custom_llm"
    
    def __init__(
        self,
        vocab_size=50257,
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        max_position_embeddings=1024,
        num_experts=4,
        expert_hidden_size=512,
        dropout=0.1,
        pad_token_id=50256,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.num_experts = num_experts
        self.expert_hidden_size = expert_hidden_size
        self.dropout = dropout
        super().__init__(pad_token_id=pad_token_id, **kwargs)

class DynamicNTKLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ntk_alpha = nn.Parameter(torch.ones(1))
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
    
    def forward(self, x):
        return x + self.ntk_alpha * torch.tanh(self.linear(x))

class ExpertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.expert_hidden_size)
        self.fc2 = nn.Linear(config.expert_hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MixtureOfExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.experts = nn.ModuleList([ExpertLayer(config) for _ in range(config.num_experts)])
        self.router = nn.Linear(config.hidden_size, config.num_experts)
        
    def forward(self, x):
        # Calculate routing weights
        route_weights = F.softmax(self.router(x), dim=-1)
        
        # Apply each expert and combine with routing weights
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        route_weights = route_weights.unsqueeze(-1)
        
        # Combine expert outputs
        output = torch.sum(expert_outputs * route_weights, dim=-2)
        return output

class CustomAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer

class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = CustomAttention(config)
        self.ntk = DynamicNTKLayer(config)
        self.experts = MixtureOfExperts(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = self.layer_norm1(hidden_states + attention_output)
        
        expert_output = self.experts(hidden_states)
        ntk_output = self.ntk(expert_output)
        hidden_states = self.layer_norm2(hidden_states + self.dropout(ntk_output))
        
        return hidden_states

class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = CustomAttention(config)
        self.cross_attention = CustomAttention(config)
        self.ntk = DynamicNTKLayer(config)
        self.experts = MixtureOfExperts(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        self.layer_norm3 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, hidden_states, encoder_hidden_states, attention_mask=None):
        self_attention_output = self.self_attention(hidden_states, attention_mask)
        hidden_states = self.layer_norm1(hidden_states + self_attention_output)
        
        cross_attention_output = self.cross_attention(hidden_states, encoder_hidden_states)
        hidden_states = self.layer_norm2(hidden_states + cross_attention_output)
        
        expert_output = self.experts(hidden_states)
        ntk_output = self.ntk(expert_output)
        hidden_states = self.layer_norm3(hidden_states + self.dropout(ntk_output))
        
        return hidden_states

class CustomLLM(PreTrainedModel):
    config_class = CustomLLMConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        self.encoder_blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.num_hidden_layers)])
        self.decoder_blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_hidden_layers)])
        
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.init_weights()
    
    def get_position_embeddings(self, position_ids):
        return self.position_embedding(position_ids)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        **kwargs
    ):
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        embeddings = self.embedding(input_ids) + self.get_position_embeddings(position_ids)
        
        # Encoder
        encoder_hidden_states = embeddings
        for encoder_block in self.encoder_blocks:
            encoder_hidden_states = encoder_block(encoder_hidden_states, attention_mask)
        
        # Decoder
        decoder_hidden_states = embeddings
        for decoder_block in self.decoder_blocks:
            decoder_hidden_states = decoder_block(decoder_hidden_states, encoder_hidden_states, attention_mask)
        
        hidden_states = self.final_layer_norm(decoder_hidden_states)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}
    
    def generate(
        self,
        input_ids,
        max_length=50,
        temperature=1.0,
        do_sample=True,
        top_k=50,
        pad_token_id=None,
        **kwargs
    ):
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        
        batch_size = input_ids.shape[0]
        
        for _ in range(max_length):
            position_ids = torch.arange(input_ids.shape[1], dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            
            outputs = self.forward(input_ids=input_ids, position_ids=position_ids)
            next_token_logits = outputs["logits"][:, -1, :] / temperature
            
            if do_sample:
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            next_tokens = next_tokens.unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            
            if next_tokens.item() == pad_token_id:
                break
        
        return input_ids

def prepare_dataset():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", clean_up_tokenization_spaces=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load a small portion of the dataset
    dataset = load_dataset("wikipedia", "20220301.en", split="train[:1%]")

    # Define a recursive text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,  # Specify the desired chunk size
        chunk_overlap=50,  # Set overlap between chunks for better context preservation
        separators=["\n\n", ".", " ", ""]
    )

    def recursive_tokenize_function(examples):
        # Split the text recursively into smaller chunks
        chunks = []
        for text in examples["text"]:
            splits = text_splitter.split_text(text)
            for split in splits:
                tokenized_split = tokenizer(split, truncation=True, padding="max_length", max_length=512)
                chunks.append(tokenized_split)
        
        return {
            "input_ids": [chunk["input_ids"] for chunk in chunks],
            "labels": [chunk["input_ids"] for chunk in chunks],  # Labels are same as input_ids for language modeling
            "attention_mask": [chunk["attention_mask"] for chunk in chunks]
        }

    # Apply the recursive tokenization
    tokenized_dataset = dataset.map(recursive_tokenize_function, batched=True, remove_columns=dataset.column_names)

    return tokenized_dataset

def train_model(model, train_dataloader, config, num_epochs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    scaler = GradScaler()
    
    wandb.init(project="custom_llm")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            # Ensure that batch contains input_ids and attention_mask in tensor format
            batch = {k: torch.tensor(v).to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                outputs = model(**batch)
                loss = outputs["loss"]
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
            wandb.log({
                "loss": loss.item(),
                "learning_rate": scheduler.get_last_lr()[0]
            })
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Average loss: {avg_loss}")
        scheduler.step()
        
        # Save checkpoint
        model.save_pretrained(f"checkpoint-epoch-{epoch+1}")
    
    wandb.finish()
    return model

def main():
    config = CustomLLMConfig()
    model = CustomLLM(config)
    
    dataset = prepare_dataset()
    train_dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    trained_model = train_model(model, train_dataloader, config)
    trained_model.save_pretrained("custom_llm_final")

if __name__ == "__main__":
    main()
