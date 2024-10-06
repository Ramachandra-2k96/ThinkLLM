import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
import os
import json
from typing import Dict, List, Tuple, Any

# Import your model classes - make sure this import path is correct
from enhanced_gpt import EnhancedCustomModelV2, EnhancedCustomConfigV2

class LocalModelTester:
    def __init__(self, model_dir: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        self.setup_logging()
        self.load_local_model()

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

    def load_local_model(self):
        try:
            self.logger.info(f"Loading model from local directory: {self.model_dir}")
            
            # Load config from local file
            config_path = os.path.join(self.model_dir, "config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at {config_path}")
            self.config = EnhancedCustomConfigV2.from_json_file(config_path)
            
            # Load model from local file
            model_path = os.path.join(self.model_dir, "pytorch_model.bin")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            self.model = EnhancedCustomModelV2(self.config)
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
            # Load tokenizer from local files
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            
            self.model.to(self.device)
            self.model.eval()
            self.logger.info("Model loaded successfully from local directory")
        except Exception as e:
            self.logger.error(f"Error loading local model: {str(e)}")
            raise

    # ... [Keep all the testing methods the same as before]
    def generate_text(self, prompt: str, max_length: int = 50) -> str:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs[0]
            
            generated = inputs.input_ids
            
            for _ in range(max_length):
                predictions = logits[:, -1, :]
                predicted_id = torch.argmax(predictions, dim=-1)
                
                if predicted_id.item() == self.tokenizer.eos_token_id:
                    break
                    
                generated = torch.cat([generated, predicted_id.unsqueeze(-1)], dim=-1)
                
                inputs = self.tokenizer(self.tokenizer.decode(generated[0]), return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs[0]
            
            return self.tokenizer.decode(generated[0], skip_special_tokens=True)
        except Exception as e:
            self.logger.error(f"Error in text generation: {str(e)}")
            return ""

    def test_attention_mechanics(self, sample_text: str) -> Dict[str, torch.Tensor]:
        try:
            inputs = self.tokenizer(sample_text, return_tensors="pt").to(self.device)
            attention_maps = {}
            
            def hook_fn(module, input, output):
                attention_maps[f"layer_{len(attention_maps)}"] = output.detach().cpu()
            
            hooks = []
            for layer in self.model.encoder:
                hooks.append(layer.attention.register_forward_hook(hook_fn))
            
            with torch.no_grad():
                self.model(**inputs)
            
            for hook in hooks:
                hook.remove()
            
            return attention_maps
        except Exception as e:
            self.logger.error(f"Error in attention mechanics test: {str(e)}")
            return {}

    def test_memory_efficiency(self) -> Dict[str, float]:
        try:
            torch.cuda.reset_peak_memory_stats()
            sample_texts = [
                "Short text.",
                "Medium length text with some more words to process.",
                "A longer text that contains multiple sentences. This should require more memory to process."
            ]
            
            memory_stats = {}
            for text in sample_texts:
                inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    self.model(**inputs)
                memory_stats[f"length_{len(text)}"] = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
                torch.cuda.reset_peak_memory_stats()
            
            return memory_stats
        except Exception as e:
            self.logger.error(f"Error in memory efficiency test: {str(e)}")
            return {}

    def test_robustness(self, text: str) -> Dict[str, Any]:
        try:
            results = {}
            
            # Test with normal input
            normal_output = self.generate_text(text)
            results['normal'] = normal_output
            
            # Test with added noise
            noisy_text = self.add_noise(text)
            noisy_output = self.generate_text(noisy_text)
            results['noisy'] = noisy_output
            
            # Test with truncated input
            truncated_text = text[:len(text)//2]
            truncated_output = self.generate_text(truncated_text)
            results['truncated'] = truncated_output
            
            return results
        except Exception as e:
            self.logger.error(f"Error in robustness test: {str(e)}")
            return {}

    @staticmethod
    def add_noise(text: str, noise_level: float = 0.1) -> str:
        chars = list(text)
        num_noise = int(len(chars) * noise_level)
        noise_positions = np.random.choice(len(chars), num_noise, replace=False)
        for pos in noise_positions:
            chars[pos] = np.random.choice(list('abcdefghijklmnopqrstuvwxyz '))
        return ''.join(chars)

    def run_all_tests(self, test_texts: List[str]) -> Dict[str, Any]:
        all_results = {}
        
        for i, text in enumerate(test_texts):
            self.logger.info(f"Running tests for text {i+1}/{len(test_texts)}")
            
            test_results = {
                'generation': self.generate_text(text),
                'attention': self.test_attention_mechanics(text),
                'robustness': self.test_robustness(text)
            }
            
            all_results[f'text_{i+1}'] = test_results
        
        memory_results = self.test_memory_efficiency()
        all_results['memory_efficiency'] = memory_results
        
        return all_results

def main():
    # Test texts
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In a world of artificial intelligence, machine learning models continue to evolve.",
        "Climate change poses significant challenges to our planet's ecosystems."
    ]
    
    # Initialize and run tests
    model_dir = "enhanced_custom_model_v2_final"  # Path to your local model directory
    tester = LocalModelTester(model_dir)
    
    results = tester.run_all_tests(test_texts)
    
    # Save results
    output_dir = "test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save text results
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump({k: v for k, v in results.items() if isinstance(v, (dict, list, str, int, float))}, f, indent=2)
    
    # Save attention visualizations
    for text_key, text_results in results.items():
        if isinstance(text_results, dict) and 'attention' in text_results:
            for layer_name, attention_map in text_results['attention'].items():
                plt.figure(figsize=(10, 8))
                plt.imshow(attention_map.mean(dim=1)[0].numpy())
                plt.title(f'{text_key} - {layer_name}')
                plt.savefig(os.path.join(output_dir, f'{text_key}_{layer_name}_attention.png'))
                plt.close()

if __name__ == "__main__":
    main()