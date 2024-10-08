import torch
from transformers import GPT2Tokenizer
import logging
logging.basicConfig(level=logging.INFO)

def load_model_and_tokenizer(model_path, tokenizer_path):
    try:
        # Load tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        tokenizer.pad_token = tokenizer.eos_token
        logging.info("Tokenizer loaded successfully")

        # Load model
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Verify checkpoint contents
        if 'model_state_dict' not in checkpoint or 'config' not in checkpoint:
            raise ValueError("Checkpoint is missing required components")
        
        config = checkpoint['config']
        
        # Import your model class
        from chatmodel import DecoderOnlyGPT
        
        # Initialize model with config
        model = DecoderOnlyGPT(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        logging.info("Model loaded successfully")
        
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading model or tokenizer: {str(e)}")
        raise

def generate_text(model, tokenizer, prompt, max_length=50):
    try:
        # Prepare input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        
        # Generate
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs['logits']
            
        # Get next token predictions
        next_token_logits = logits[0, -1, :]
        next_token = torch.argmax(next_token_logits).item()
        
        # Decode
        generated_text = tokenizer.decode(inputs['input_ids'][0]) + tokenizer.decode([next_token])
        return generated_text
    except Exception as e:
        logging.error(f"Error during text generation: {str(e)}")
        return None

def test_model():
    MODEL_PATH = "enhanced_wikipedia_chatbot_model.pt"
    TOKENIZER_PATH = "enhanced_wikipedia_chatbot_tokenizer"
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(MODEL_PATH, TOKENIZER_PATH)
        
        # Test prompts
        test_prompts = [
            "The capital of France is",
            "Artificial Intelligence is",
            "The purpose of education is"
        ]
        
        # Generate and print responses
        for prompt in test_prompts:
            logging.info(f"\nPrompt: {prompt}")
            response = generate_text(model, tokenizer, prompt)
            if response:
                logging.info(f"Generated: {response}")
            else:
                logging.warning(f"Failed to generate response for: {prompt}")
                
        # Test memory states
        logging.info("\nTesting memory states...")
        outputs = model(
            **tokenizer("This is a test of memory states.", return_tensors="pt"),
            output_hidden_states=True
        )
        if 'memory_states' in outputs:
            logging.info("Memory states successfully generated")
            
        # Test model components
        logging.info("\nVerifying model components...")
        expected_attributes = ['cognitive', 'ntk', 'decision_trees']
        for block in model.h:
            for attr in expected_attributes:
                if hasattr(block, attr):
                    logging.info(f"{attr} component found in model block")
                else:
                    logging.warning(f"{attr} component not found in model block")
    
    except Exception as e:
        logging.error(f"Test failed: {str(e)}")

if __name__ == "__main__":
    logging.info("Starting model test...")
    test_model()