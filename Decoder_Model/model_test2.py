import torch
from transformers import GPT2Tokenizer
from chatmodel import DecoderOnlyGPT, DecoderOnlyConfig, load_enhanced_model

def test_model(model_path, prompts):
    # Load the model and tokenizer
    model, tokenizer = load_enhanced_model(model_path, training=False)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("Generated text:")
        
        # Tokenize the prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Generate text
        with torch.no_grad():
            output = model.generate(
                inputs=input_ids,
                max_length=100,
                temperature=1.0, top_k=50, top_p=0.95
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(generated_text)

if __name__ == "__main__":
    # Path to the saved model
    model_path = "final_enhanced_wikipedia_chatbot"
    
    # List of prompts to test
    prompts = [
        "The capital of France is",
        "Artificial intelligence is",
        "The theory of relativity states that",
        "In computer science, a neural network is",
        "The process of photosynthesis involves"
    ]
    
    # Test the model
    test_model(model_path, prompts)