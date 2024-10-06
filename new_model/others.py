import os
import torch
import json
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup
from torch.amp import GradScaler,autocast
import gc

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