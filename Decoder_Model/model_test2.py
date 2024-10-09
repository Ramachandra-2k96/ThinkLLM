import torch
from transformers import GPT2Tokenizer
from typing import List

# Import the necessary classes and functions
from chatmodel import DecoderOnlyGPT, DecoderOnlyConfig, load_enhanced_model

def load_model_for_chat(model_path: str):
    """
    Load the saved model and prepare it for conversation.
    """
    # Load the model, tokenizer, and other components
    model, tokenizer = load_enhanced_model(model_path, training=False)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode
    
    return model, tokenizer, device

def generate_response(model: DecoderOnlyGPT, tokenizer: GPT2Tokenizer, 
                      input_text: str, device: torch.device, 
                      max_length: int = 100) -> str:
    """
    Generate a response using the loaded model.
    """
    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    
    # Generate output
    with torch.no_grad():
        generated_ids_beam = model.generate(
            input_ids,
            num_beams=4,
            num_beam_groups=2,
            diversity_penalty=1.0,
            length_penalty=1.0,
            decoder_strategy="beam_search",
            cognitive_memory_influence=0.3,
            expert_confidence_threshold=0.7,
            max_length=200
        )

        # Greedy
        generated_ids_greedy = model.generate(
            input_ids,
            decoder_strategy="greedy",
            cognitive_memory_influence=0.3,
            expert_confidence_threshold=0.7,
            max_length=200
        )

        # Sampling
        generated_ids_sample = model.generate(
            input_ids,
            decoder_strategy="sample",
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            cognitive_memory_influence=0.3,
            expert_confidence_threshold=0.7,
            max_length=200
        )

    # Decode generated sequences
    beam_text = tokenizer.decode(generated_ids_beam[0], skip_special_tokens=True)
    greedy_text = tokenizer.decode(generated_ids_greedy[0], skip_special_tokens=True)
    sample_text = tokenizer.decode(generated_ids_sample[0], skip_special_tokens=True)
    # Decode the output
    response = beam_text + " \n\n"+greedy_text+"\n\n"+sample_text
    
    return response

def chat_loop(model: DecoderOnlyGPT, tokenizer: GPT2Tokenizer, device: torch.device):
    """
    Run an interactive chat loop with the model.
    """
    print("Chat with the AI (type 'quit' to exit):")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        
        response = generate_response(model, tokenizer, user_input, device)
        print("AI:", response)

def main():
    model_path = "checkpoint_epoch_2"  # Update this to your model's path
    model, tokenizer, device = load_model_for_chat(model_path)
    chat_loop(model, tokenizer, device)

if __name__ == "__main__":
    main()