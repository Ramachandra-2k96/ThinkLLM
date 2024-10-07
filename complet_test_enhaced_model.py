import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import numpy as np
import logging
import os
import json
from typing import Dict, List, Any

# Import your model classes
from enhanced_gpt import EnhancedCustomModelV2, EnhancedCustomConfigV2

class ModelTester:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.setup_logging()
        self.load_model()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('model_test.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_model(self):
        try:
            self.logger.info(f"Loading model from {self.model_path}")
            self.config = EnhancedCustomConfigV2.from_pretrained(self.model_path)
            self.model = EnhancedCustomModelV2.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, clean_up_tokenization_spaces=True)
            self.model.to(self.device)
            self.model.eval()
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def generate_text(self, prompt: str, max_length: int = 500, repetition_penalty: float = 1.6) -> str:
        try:
            # Tokenize input prompt
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
            
            # Generate text
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_length,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.3,
                    top_k = 50,
                )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            return generated_text
        except Exception as e:
            self.logger.error(f"Error in text generation: {str(e)}")
            return ""

    def run_all_tests(self, test_texts: List[str]) -> Dict[str, Any]:
        all_results = {}
        
        for i, text in enumerate(test_texts):
            self.logger.info(f"Running tests for text {i+1}/{len(test_texts)}")
            
            generated_text = self.generate_text(text)
            all_results[f'text_{i+1}'] = {
                'original': text,
                'generated': generated_text
            }
        
        return all_results

    def save_results(self, results: Dict[str, Any], output_path: str):
        try:
            os.makedirs(output_path, exist_ok=True)
            
            # Save text results
            with open(os.path.join(output_path, 'test_results.json'), 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"Results saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")

def main():
    # Test texts
    test_texts = [
        "The quick",
        "In a world of artificial intelligence, machine",
        "Climate change poses significant challenges to"
    ]
    
    # Initialize and run tests
    model_path = "enhanced_custom_model_v2_final" 
    tester = ModelTester(model_path)
    
    results = tester.run_all_tests(test_texts)
    
    # Save results
    tester.save_results(results, "test_results")

if __name__ == "__main__":
    main()
