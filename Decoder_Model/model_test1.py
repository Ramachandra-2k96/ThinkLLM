import torch
from transformers import GPT2Tokenizer

from Decoder_Model.chatmodel import DecoderOnlyGPT
# Load the trained model and tokenizer
model_path = "enhanced_wikipedia_chatbot_tokenizer"  # Update this to your model path
tokenizer_path = "enhanced_wikipedia_chatbot_tokenizer"  # Update this to your tokenizer path

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
model = DecoderOnlyGPT.from_pretrained(model_path)

# Set the model to evaluation mode
model.eval()

# Function to generate a response from the model
def generate_response(prompt, max_length=50, temperature=1.0):
    # Encode the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate a response
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, temperature=temperature, pad_token_id=tokenizer.eos_token_id)
    
    # Decode the output tokens to string
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Chat loop
print("Chatbot is ready! Type 'exit' to stop the chat.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    # Generate response
    response = generate_response(user_input)
    print("Chatbot:", response)
