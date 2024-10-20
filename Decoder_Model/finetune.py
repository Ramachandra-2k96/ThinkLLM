import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_cosine_schedule_with_warmup
from datasets import load_dataset
from tqdm.auto import tqdm
import os

# Import your model and config classes
from ThinkLLM import EnhancedDecoderOnlyGPT, EnhancedDecoderOnlyConfig, load_enhanced_model, save_enhanced_model

class DailyDialogDataset(Dataset):
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load DailyDialog dataset
        dataset = load_dataset("daily_dialog")
        self.dialogs = dataset["train"]["dialog"]
        
    def __len__(self):
        return len(self.dialogs)
    
    def __getitem__(self, idx):
        dialog = self.dialogs[idx]
        
        # Combine all utterances in the dialog
        full_dialog = " ".join(dialog)
        
        # Tokenize the dialog
        inputs = self.tokenizer(full_dialog, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        
        # Create labels (shifted input_ids for language modeling)
        labels = inputs['input_ids'].clone()
        labels[:, :-1] = inputs['input_ids'][:, 1:]
        labels[:, -1] = -100  # Ignore prediction for the last token
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }

def fine_tune_chatbot(model, tokenizer, dataset, config, num_epochs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    scaler = torch.amp.GradScaler()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with torch.amp.autocast('cuda'):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs['loss']
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        save_enhanced_model(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            tokenizer=tokenizer,
            config=config,
            epoch=epoch+1,
            train_loss_history=[avg_loss],
            scaler=scaler,
            path=f"fine_tuned_chatbot_checkpoint_epoch_{epoch+1}"
        )
    
    return model

def main():
    # Load your pre-trained model
    model, tokenizer = load_enhanced_model("results/checkpoint_epoch_2")
    
    # Update the config for fine-tuning
    config = model.config
    config.learning_rate = 5e-5  # Lower learning rate for fine-tuning
    config.batch_size = 4
    config.warmup_ratio = 0.1
    config.max_grad_norm = 1.0
    
    # Prepare the DailyDialog dataset
    dataset = DailyDialogDataset(tokenizer, max_length=config.n_positions)
    
    # Fine-tune the model
    fine_tuned_model = fine_tune_chatbot(model, tokenizer, dataset, config, num_epochs=3)
    
    # Save the final fine-tuned model
    save_enhanced_model(
        model=fine_tuned_model,
        optimizer=None,
        scheduler=None,
        tokenizer=tokenizer,
        config=config,
        epoch=3,
        train_loss_history=None,
        scaler=None,
        path="final_fine_tuned_chatbot"
    )
    
    print("Fine-tuning completed and model saved.")

if __name__ == "__main__":
    main()