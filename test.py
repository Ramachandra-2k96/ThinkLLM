import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import math
import os
import json
import warnings

# Suppress FutureWarning for clean_up_tokenization_spaces
warnings.filterwarnings("ignore", category=FutureWarning)

# Import the custom classes from the previous implementation
from new import EnhancedCustomModel, EnhancedCustomConfig, WikipediaDataset

def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0):
    model.eval()
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_logits = outputs['logits'][:, -1, :] / temperature
            next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(input_ids[0], clean_up_tokenization_spaces=True)

def calculate_perplexity(model, data_loader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
            
            total_loss += loss.item() * input_ids.size(1)
            total_tokens += input_ids.size(1)
    
    perplexity = math.exp(total_loss / total_tokens)
    return perplexity

def load_enhanced_custom_model(model_path):
    config = EnhancedCustomConfig.from_pretrained(model_path)
    model = EnhancedCustomModel(config)
    
    state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    
    custom_components = torch.load(os.path.join(model_path, "custom_components.pt"), map_location=torch.device('cpu'), weights_only=True)
    for i, layer in enumerate(model.layers):
        layer[1].load_state_dict(custom_components["tree_layers"][i])
        layer[2].load_state_dict(custom_components["thought_layers"][i])
    
    tied_weights_info_path = os.path.join(model_path, "tied_weights_info.json")
    if os.path.exists(tied_weights_info_path):
        with open(tied_weights_info_path, "r") as f:
            tied_weights_info = json.load(f)
        if tied_weights_info.get("lm_head_tied_to_embeddings", False):
            model.lm_head.weight = model.word_embeddings.weight
    
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", clean_up_tokenization_spaces=True)
    model = load_enhanced_custom_model("enhanced_custom_model_v2_final")
    model.to(device)

    # Test text generation
    prompt = "The history of artificial intelligence"
    generated_text = generate_text(model, tokenizer, prompt)
    print(f"Generated text:\n{generated_text}\n")

    # Prepare test dataset
    test_dataset = WikipediaDataset(tokenizer, max_length=128, subset_size=10)
    test_loader = DataLoader(test_dataset, batch_size=4, pin_memory=True)

    # Calculate perplexity
    perplexity = calculate_perplexity(model, test_loader, device)
    print(f"Model perplexity: {perplexity:.2f}")

if __name__ == "__main__":
    main()