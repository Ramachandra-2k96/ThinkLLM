import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm.auto import tqdm
import random
import numpy as np
from ThinkLLM import EnhancedDecoderOnlyGPT, EnhancedDecoderOnlyConfig, load_enhanced_model, save_enhanced_model
import matplotlib.pyplot as plt
import os
import wandb
from torch.utils.data import Dataset
from datasets import load_dataset

class ConversationDataset(Dataset):
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load dataset
        dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
        
        # Let's first print a sample to debug
        print("Dataset sample:", dataset[0])
        
        # Process conversations
        self.conversations = []
        for item in dataset:
            # The dataset structure is different - it uses 'text' field
            # which contains the full conversation
            conversation_text = item['text']
            
            # Split into human and assistant parts
            # The format is typically "Human: ... Assistant: ..."
            parts = conversation_text.split("Assistant:", 1)
            if len(parts) != 2:
                continue  # Skip malformed conversations
                
            human_part = parts[0]
            assistant_part = parts[1]
            
            # Clean up and format
            human_msg = f"### Human: {human_part.replace('Human:', '').strip()}"
            assistant_msg = f"### Assistant: {assistant_part.strip()}"
            
            self.conversations.append({
                'human': human_msg,
                'assistant': assistant_msg
            })
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        
        # Combine messages with proper formatting
        full_text = f"{conversation['human']}\n{conversation['assistant']}"
        
        # Tokenize the full conversation
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors="pt"
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Create labels for causal language modeling
        labels = input_ids.clone()
        
        # Mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # Find the assistant's response start to mask human input
        response_start = full_text.find("### Assistant:")
        if response_start != -1:
            # Tokenize everything before the assistant's response
            human_tokens = self.tokenizer(
                full_text[:response_start], 
                return_tensors="pt", 
                add_special_tokens=True
            )
            # Mask out the human's part in labels
            human_length = human_tokens['input_ids'].size(1)
            labels[:human_length] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
class TrainingConfig:
    def __init__(self):
        self.learning_rate = 2e-5
        self.batch_size = 1
        self.gradient_accumulation_steps = 4
        self.warmup_ratio = 0.1
        self.max_grad_norm = 1.0
        self.weight_decay = 0.01
        self.num_epochs = 3
        self.patience = 2
        self.evaluation_steps = 100
        self.logging_steps = 10
        self.save_steps = 500

def train_chatbot(model, tokenizer, train_dataset, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Initialize wandb for better training monitoring
    wandb.init(project="chatbot-training", config=vars(config))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,  # Reduced number of workers for stability
        pin_memory=True
    )
    
    # Initialize optimizer with weight decay for regularization
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
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    
    # Setup learning rate scheduler
    num_training_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * config.warmup_ratio),
        num_training_steps=num_training_steps
    )
    
    scaler = torch.amp.GradScaler()
    best_loss = float('inf')
    patience_counter = 0
    global_step = 0
    
    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)
    
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            try:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass with gradient accumulation
                with torch.amp.autocast('cuda'):
                    outputs = model(**batch)
                    loss = outputs['loss'] / config.gradient_accumulation_steps
                
                # Backward pass
                scaler.scale(loss).backward()
                
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        config.max_grad_norm
                    )
                    
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                    
                    # Log metrics
                    if global_step % config.logging_steps == 0:
                        wandb.log({
                            "loss": loss.item() * config.gradient_accumulation_steps,
                            "learning_rate": scheduler.get_last_lr()[0],
                            "global_step": global_step
                        })
                    
                    # Save checkpoint
                    if global_step % config.save_steps == 0:
                        save_enhanced_model(
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            tokenizer=tokenizer,
                            config=model.config,
                            epoch=epoch,
                            train_loss_history=[loss.item()],
                            scaler=scaler,
                            path=f"checkpoints/step_{global_step}"
                        )
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": loss.item() * config.gradient_accumulation_steps,
                    "step": global_step
                })
                
                total_loss += loss.item()
                
            except Exception as e:
                print(f"Error in training loop: {str(e)}")
                continue
        
        # Evaluate epoch performance
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model
            save_enhanced_model(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                tokenizer=tokenizer,
                config=model.config,
                epoch=epoch,
                train_loss_history=[avg_loss],
                scaler=scaler,
                path="best_model"
            )
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print("Early stopping triggered")
                break
    
    wandb.finish()
    return model,optimizer,scheduler,[avg_loss],scaler

def main():
    # Set random seeds
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_enhanced_model("results/checkpoint_epoch_2")
        
        # Initialize training config
        config = TrainingConfig()
        
        # Create dataset
        dataset = ConversationDataset(tokenizer, max_length=model.config.n_positions)
        
        # Train model
        fine_tuned_model,optimizer,scheduler,train_loss_history,scaler= train_chatbot(model, tokenizer, dataset, config)
        
        # Save final model without optimizer state
        save_enhanced_model(
            model=fine_tuned_model,
            optimizer=optimizer,  # Explicitly None for final save
            scheduler=scheduler,
            tokenizer=tokenizer,
            config=model.config,
            epoch=config.num_epochs,
            train_loss_history=train_loss_history,
            scaler=scaler,
            path="final_model"
        )
        
        print("Fine-tuning completed successfully.")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()