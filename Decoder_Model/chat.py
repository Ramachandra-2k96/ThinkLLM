import torch
from transformers import AutoTokenizer
from ThinkLLM import load_enhanced_model
import torch.nn.functional as F

class Chatbot:
    def __init__(self, model_path="final_model"):
        """Initialize the chatbot with the fine-tuned model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from {model_path}...")
        self.model, self.tokenizer = load_enhanced_model(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.conversation_history = []
        
    def generate_response(self, user_input, max_length=100, temperature=0.7, top_p=0.9):
        """Generate a response to the user input using manual token generation."""
        # Add user input to conversation history
        self.conversation_history.append(f"User: {user_input}")
        
        # Prepare context from conversation history
        context = " ".join(self.conversation_history[-3:])  # Use last 3 turns for context
        
        # Tokenize input
        input_ids = self.tokenizer.encode(context, return_tensors="pt").to(self.device)
        attention_mask = torch.ones_like(input_ids).to(self.device)
        
        # Generate response token by token
        for _ in range(max_length):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                
                next_token_logits = outputs["logits"][:, -1, :] / temperature
                
                # Apply top-p (nucleus) sampling
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Stop if we predict the end of sequence token
                if next_token[0].item() == self.tokenizer.eos_token_id:
                    break
                
                # Concatenate next token to input_ids
                input_ids = torch.cat([input_ids, next_token], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token).to(self.device)], dim=1)
        
        # Decode the generated sequence
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        # Extract only the generated response (remove the input context)
        response = generated_text[len(context):].strip()
        
        # Add response to conversation history
        self.conversation_history.append(f"Assistant: {response}")
        
        return response
    
    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = []

def main():
    # Initialize chatbot
    print("Initializing chatbot...")
    chatbot = Chatbot()
    print("Chatbot is ready! Type 'quit' to exit or 'reset' to clear conversation history.")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        elif user_input.lower() == 'reset':
            chatbot.reset_conversation()
            print("Conversation history cleared!")
            continue
        elif not user_input:
            continue
            
        try:
            response = chatbot.generate_response(user_input)
            print(f"\nAssistant: {response}")
        except Exception as e:
            print(f"Error generating response: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()