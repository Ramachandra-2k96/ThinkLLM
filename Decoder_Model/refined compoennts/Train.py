from save_load import save_enhanced_model
from transformers import get_cosine_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

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