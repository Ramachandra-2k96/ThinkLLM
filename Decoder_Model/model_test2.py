import torch
import argparse
from transformers import GPT2Tokenizer
from chatmodel import DecoderOnlyGPT, DecoderOnlyConfig
import os

def load_model_for_inference(model_path):
    """
    Load the saved model for inference, including all enhanced components.
    """
    # Load config and create model
    config = DecoderOnlyConfig.from_pretrained(model_path)
    model = DecoderOnlyGPT(config)
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_path,)
    
    # Load the full state dictionary
    full_state_path = os.path.join(model_path, 'training_state.bin')
    if os.path.exists(full_state_path):
        full_state = torch.load(full_state_path, map_location=torch.device('cpu'))
        
        # Load the main model weights
        model.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location=torch.device('cpu')))
        
        # Load additional components
        additional_components = full_state.get('additional_components', {})
        
        for i, block in enumerate(model.h):
            if hasattr(block.mlp, 'experts') and 'moe_states' in additional_components:
                block.mlp.load_state_dict(additional_components['moe_states'][i])
            if hasattr(block, 'ntk') and 'ntk_states' in additional_components:
                block.ntk.load_state_dict(additional_components['ntk_states'][i])
            if hasattr(block, 'decision_trees') and 'decision_tree_states' in additional_components:
                block.decision_trees.load_state_dict(additional_components['decision_tree_states'][i])
            if hasattr(block, 'cognitive') and 'cognitive_states' in additional_components:
                block.cognitive.load_state_dict(additional_components['cognitive_states'][i])
        
        # Load memory states if they exist
        if 'memory_states' in full_state.get('training_state', {}):
            for i, block in enumerate(model.h):
                if hasattr(block, 'cognitive'):
                    block.cognitive.memory_state = full_state['training_state']['memory_states'][i]
    else:
        print(f"Warning: Full state file not found at {full_state_path}. Loading only the base model.")
        model = DecoderOnlyGPT.from_pretrained(model_path, config=config)
    
    return model, tokenizer
def generate_response(model, tokenizer, input_text, max_length=100):
    """
    Generate a response from the model given an input text.
    """
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    # Move input to the same device as the model
    input_ids = input_ids.to(model.device)
    
    # Set the pad token ID to the EOS token ID if not set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Generate output
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode the output
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return response

def chat_loop(model, tokenizer):
    """
    Run an interactive chat loop with the model.
    """
    print("Welcome to the Enhanced Wikipedia Chatbot!")
    print("Type 'quit' or 'exit' to end the conversation.")
    
    context = ""
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        # Append user input to context
        context += f"Human: {user_input}\nAI: "
        
        # Generate response
        response = generate_response(model, tokenizer, context)
        
        # Extract the AI's response
        ai_response = response.split("AI: ")[-1].strip()
        
        print(f"AI: {ai_response}")
        
        # Update context
        context += f"{ai_response}\n"

def main():
    model_path = "final_enhanced_wikipedia_chatbot"
    # Load model and tokenizer
    print(f"Loading model from {model_path}")
    model, tokenizer = load_model_for_inference(model_path)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode

    print(f"Model loaded successfully. Using device: {device}")

    # Start chat loop
    chat_loop(model, tokenizer)

if __name__ == "__main__":
    main()