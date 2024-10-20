import torch
import torch.nn.functional as F
from ThinkLLM import load_enhanced_model
import torch
import torch.nn.functional as F

def generate_text(model, tokenizer, prompt, max_length=500, temperature=0.75, top_k=50):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    
    with torch.no_grad():
        for _ in range(max_length - input_ids.shape[1]):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            next_token_logits = outputs["logits"][:, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Filter with top-k
            top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
            
            # Apply softmax to convert logits to probabilities
            probs = F.softmax(top_k_logits, dim=-1)
            
            # Check for NaN or inf values
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                print("Warning: NaN or inf detected in probabilities. Applying fallback strategy.")
                probs = torch.ones_like(probs) / probs.size(-1)
            
            # Ensure probabilities sum to 1
            probs = probs / probs.sum(dim=-1, keepdim=True)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices.gather(-1, next_token)
            
            # Concatenate next token to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
            
            # Check if EOS token was generated
            if next_token[0, 0].item() == tokenizer.eos_token_id:
                break
    
    # Decode the generated sequence
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text

def test_generation(model_path="results/checkpoint_epoch_2"):
    # Load the model and tokenizer
    model, tokenizer = load_enhanced_model(model_path)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Test prompts
    test_prompts = [
        "The history of artificial intelligence",
        "Climate change impacts on",
        "The role of technology in modern education",
    ]
    
    # Generate and print results
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 50)
        generated_text = generate_text(model, tokenizer, prompt)
        print(f"Generated text: {generated_text}")
        print("-" * 50)

if __name__ == "__main__":
    test_generation()