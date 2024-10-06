import torch
from transformers import AutoTokenizer
from enhanced_gpt import EnhancedCustomModelV2, EnhancedCustomConfigV2
import os
import json
from typing import List, Dict

class ModelTester:
    def __init__(self, model_dir: str = "enhanced_custom_model_v2_final"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        self.load_model()
        
    def load_model(self):
        # Load config
        with open(os.path.join(self.model_dir, "config.json"), "r") as f:
            config_dict = json.load(f)
        self.config = EnhancedCustomConfigV2(**config_dict)
        
        # Load model
        self.model = EnhancedCustomModelV2(self.config)
        model_path = os.path.join(self.model_dir, "pytorch_model.bin")
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        
        print(f"Model loaded successfully to {self.device}")

    def prepare_input(self, text: str) -> Dict[str, torch.Tensor]:
        # Tokenize input text
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.config.max_position_embeddings,
            return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    @torch.no_grad()
    def generate_text(self, prompt: str, max_new_tokens: int = 50) -> str:
        inputs = self.prepare_input(prompt)
        
        # Generate tokens
        generated = inputs["input_ids"]
        past_key_values = None
        
        for _ in range(max_new_tokens):
            outputs = self.model(
                input_ids=generated[:, -1:],
                attention_mask=torch.ones(1, 1, device=self.device),
                use_cache=True,
                past_key_values=past_key_values
            )
            
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs[0]
            past_key_values = outputs.get('past_key_values', None)
            
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
                
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
        
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

    def test_model(self, test_prompts: List[str]):
        print("\nRunning model tests...")
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\nTest {i}:")
            print(f"Prompt: {prompt}")
            
            try:
                # Try generating text
                generated_text = self.generate_text(prompt)
                print(f"Generated: {generated_text}")
                
                # Get model outputs for analysis
                inputs = self.prepare_input(prompt)
                outputs = self.model(**inputs)
                
                # Print some statistics
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                    print(f"Output logits shape: {logits.shape}")
                    print(f"Max logit value: {logits.max().item():.4f}")
                    print(f"Min logit value: {logits.min().item():.4f}")
                
            except Exception as e:
                print(f"Error during generation: {str(e)}")

def main():
    # Test prompts
    test_prompts = [
        "Once upon a time",
        "The future of artificial intelligence",
        "In the world of science",
        "A simple recipe for"
    ]
    
    try:
        tester = ModelTester()
        tester.test_model(test_prompts)
    except Exception as e:
        print(f"Error during testing: {str(e)}")

if __name__ == "__main__":
    main()