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

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=len(train_loader) * config.epochs)
    
    scaler = GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]")
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(train_pbar):
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)

            try:
                with autocast(device_type=device.type, dtype=torch.float16):
                    outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
                    loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
                    
                    if not torch.isfinite(loss):
                        print(f"Warning: Non-finite loss detected: {loss.item()}. Skipping batch.")
                        continue
                    
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

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"CUDA out of memory in batch {i}. Skipping batch and clearing cache.")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            del input_ids, attention_mask, outputs, loss
            torch.cuda.empty_cache()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{config.epochs}, Train Loss: {avg_loss:.4f}")

        gc.collect()
        torch.cuda.empty_cache()

    return model