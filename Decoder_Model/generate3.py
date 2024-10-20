import torch
from ThinkLLM2 import EnhancedDecoderOnlyGPT, GPT2Tokenizer 

checkpoint_dir = 'checkpoints/checkpoint-1500'
tokenizer = GPT2Tokenizer.from_pretrained('gpt2',clean_up_tokenization_spaces=True)
tokenizer.pad_token = tokenizer.eos_token
from transformers import PretrainedConfig

config = PretrainedConfig.from_json_file(f"{checkpoint_dir}/config.json")

model = EnhancedDecoderOnlyGPT(config)
model.load_state_dict(torch.load(f"{checkpoint_dir}/pytorch_model.bin", map_location=torch.device('cpu'),weights_only=False), strict=False)

model.eval()
if model.config.pad_token_id is None:
    model.config.pad_token_id = model.config.eos_token_id

def generate_text(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors='pt', padding=True)
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], attention_mask=attention_mask, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

# Example usage
prompt = "Once upon a time"
generated_text = generate_text(prompt)
print(generated_text)
