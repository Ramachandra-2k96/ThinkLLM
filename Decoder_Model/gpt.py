import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig

class DecoderOnlyConfig(PretrainedConfig):
    model_type = "decoder_only_gpt"

    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        tie_word_embeddings=True,
        use_moe=False,
        num_experts=4,
        top_k_experts=2,
        **kwargs
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts

class DecoderOnlyAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.attn_pdrop

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _attn(self, query, key, value, attention_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        attn_weights = attn_weights / torch.sqrt(torch.tensor(value.size(-1), dtype=torch.float32))
        
        if attention_mask is not None:
            # Ensure the attention_mask has the right shape
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights

    def forward(self, hidden_states, attention_mask=None):
        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(self.n_embd, dim=2)
        
        query = query.view(*query.size()[:-1], self.n_head, -1).transpose(1, 2)
        key = key.view(*key.size()[:-1], self.n_head, -1).transpose(1, 2)
        value = value.view(*value.size()[:-1], self.n_head, -1).transpose(1, 2)

        attn_output, _ = self._attn(query, key, value, attention_mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(*hidden_states.size()[:-1], self.n_embd)
        
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        return attn_output

class DecoderOnlyMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_inner)
        self.c_proj = nn.Linear(config.n_inner, config.n_embd)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class DecoderOnlyBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = DecoderOnlyAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = DecoderOnlyMLP(config)

    def forward(self, hidden_states, attention_mask=None):
        attn_output = self.attn(self.ln_1(hidden_states), attention_mask=attention_mask)
        hidden_states = hidden_states + attn_output
        mlp_output = self.mlp(self.ln_2(hidden_states))
        hidden_states = hidden_states + mlp_output
        return hidden_states

class MixtureOfExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        self.experts = nn.ModuleList([DecoderOnlyMLP(config) for _ in range(self.num_experts)])
        self.gate = nn.Linear(config.n_embd, self.num_experts)

    def forward(self, hidden_states):
        expert_weights = F.softmax(self.gate(hidden_states), dim=-1)
        top_k_weights, top_k_indices = torch.topk(expert_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        output = torch.zeros_like(hidden_states)
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, :, k]
            expert_weight = top_k_weights[:, :, k].unsqueeze(-1)
            for i in range(self.num_experts):
                expert_mask = (expert_idx == i)
                if expert_mask.any():
                    expert_input = hidden_states[expert_mask]
                    expert_output = self.experts[i](expert_input)
                    output[expert_mask] += expert_weight[expert_mask] * expert_output
        
        return output

class DecoderOnlyGPT(PreTrainedModel):
    config_class = DecoderOnlyConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([DecoderOnlyBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        if config.use_moe:
            self.moe = MixtureOfExperts(config)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.wte.weight

        self.init_weights()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        if position_ids is None:
            position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=inputs_embeds.device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        for block in self.h:
            hidden_states = block(hidden_states, attention_mask)

        hidden_states = self.ln_f(hidden_states)

        if self.config.use_moe:
            hidden_states = self.moe(hidden_states)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {
            "loss": loss,
            "logits": lm_logits,
            "hidden_states": hidden_states,
        }

    def generate(
        self,
        input_ids,
        max_length=50,
        min_length=10,
        do_sample=True,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        pad_token_id=None,
        eos_token_id=None,
        attention_mask=None,
        num_return_sequences=1,
        no_repeat_ngram_size=None
    ):
        # Set the model to evaluation mode
        self.eval()
        
        # Move input_ids to the same device as the model
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Repeat input IDs for num_return_sequences
        batch_size = input_ids.size(0)
        input_ids = input_ids.repeat_interleave(num_return_sequences, dim=0)
        
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(num_return_sequences, dim=0)

        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=device)
        
        # To track generated n-grams if no_repeat_ngram_size is set
        generated_ngrams = [{} for _ in range(input_ids.size(0))] if no_repeat_ngram_size else None
        
        with torch.no_grad():
            while input_ids.shape[-1] < max_length:
                # Forward pass
                outputs = self(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                
                next_token_logits = outputs["logits"][:, -1, :] / temperature
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(input_ids.shape[0]):
                        for previous_token in set(input_ids[i].tolist()):
                            next_token_logits[i, previous_token] /= repetition_penalty
                
                # Filter using top-k and top-p
                if do_sample:
                    # Top-k filtering
                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Top-p filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Sample from filtered distribution
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Check n-gram repetition
                if no_repeat_ngram_size:
                    for i, (tokens, ngrams) in enumerate(zip(input_ids, generated_ngrams)):
                        generated_ngrams[i] = self.update_ngram_constraints(
                            ngrams, tokens, next_tokens[i], no_repeat_ngram_size
                        )
                
                # Update input_ids and attention_mask
                input_ids = torch.cat([input_ids, next_tokens], dim=-1)
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        attention_mask.new_ones((attention_mask.shape[0], 1))
                    ], dim=-1)
                
                # Check if sequences are finished
                if eos_token_id is not None:
                    unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())
                
                # Stop if all sequences are finished or max_length is reached
                if unfinished_sequences.max() == 0 or input_ids.shape[-1] >= max_length:
                    break
        
        # Reshape output if num_return_sequences > 1
        if num_return_sequences > 1:
            return input_ids.view(batch_size, num_return_sequences, -1)
        
        return input_ids

    def update_ngram_constraints(self, ngrams, input_seq, next_token, n):
        """Helper function to update n-gram tracking for no_repeat_ngram_size."""
        if len(input_seq) >= n - 1:
            gram = tuple(input_seq[-(n-1):].tolist())
            if gram in ngrams and next_token.item() in ngrams[gram]:
                # Find the next best token that doesn't create a repeated n-gram
                logits = self(input_ids=input_seq.unsqueeze(0))["logits"][0, -1]
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                for new_token in sorted_indices:
                    if gram not in ngrams or new_token.item() not in ngrams[gram]:
                        next_token.fill_(new_token.item())
                        break
            ngrams.setdefault(gram, set()).add(next_token.item())
        return ngrams

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from tqdm.auto import tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter

class WikipediaDataset(Dataset):
    def __init__(self, tokenizer, max_length=512, num_examples=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load Wikipedia dataset
        dataset = load_dataset("wikipedia", "20220301.en", split="train[:1%]")
        
        # Use RecursiveCharacterTextSplitter from LangChain
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        self.texts = []
        for item in tqdm(dataset, desc="Processing Wikipedia articles"):
            chunks = text_splitter.split_text(item['text'])
            self.texts.extend(chunks)
        
        # Limit the number of examples if specified
        if num_examples is not None:
            self.texts = self.texts[:num_examples]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize the text
        inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        
        # Create labels (shifted input_ids)
        labels = inputs['input_ids'].clone()
        labels[:, :-1] = inputs['input_ids'][:, 1:]
        labels[:, -1] = -100  # Ignore the last token prediction
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

def train_chatbot_model(model, dataset, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # If CUDA is available but there's not enough memory, fall back to CPU
    if device.type == "cuda":
        try:
            torch.cuda.empty_cache()
            model = model.to(device)
        except RuntimeError:
            print("Not enough GPU memory. Falling back to CPU.")
            device = torch.device("cpu")
            model = model.to(device)
    else:
        model = model.to(device)

    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    
    # Learning rate scheduler
    num_training_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=config.warmup_steps, 
        num_training_steps=num_training_steps
    )

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
            # Update progress bar
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / (progress_bar.n + 1):.4f}",
                'lr': f"{current_lr:.2e}"
            })
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{config.num_epochs}, Average Loss: {avg_loss:.4f}, Final LR: {current_lr:.2e}")

    return model

def generate_response(model, tokenizer, input_text, max_length=100):
    model.eval()  # Set the model to evaluation mode
    input_ids = tokenizer.encode(input_text, return_tensors="pt")  # Tokenize the input

    with torch.no_grad():  # Disable gradient computation
        # Generate text using the updated generate method
        output = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_return_sequences=1,  # Set how many sequences to generate
            no_repeat_ngram_size=2,  # Avoid repeating n-grams
            do_sample=True,  # Enable sampling
            temperature=0.7,  # Control randomness
            top_k=50,  # Limit to top-k tokens
            top_p=0.95  # Use nucleus sampling
        )

    # Decode the generated token IDs to text
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


def main(num_examples=None):
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2",clean_up_tokenization_spaces=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize model configuration
    config = DecoderOnlyConfig(
        vocab_size=tokenizer.vocab_size,
        n_positions=512,
        n_embd=768,
        n_layer=8,
        n_head=8,
        n_inner=3072-768,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_moe=True,
        num_experts=4,
        top_k_experts=2
    )

    # Create model
    model = DecoderOnlyGPT(config)

    # Create dataset
    dataset = WikipediaDataset(tokenizer, num_examples=num_examples)

    # Training configuration
    train_config = type('TrainConfig', (), {
        'batch_size': 2,
        'learning_rate': 5e-5,
        'num_epochs': 3,
        'warmup_steps': 100
    })()

    # Train the model
    trained_model = train_chatbot_model(model, dataset, train_config)

    # Save the trained model
    trained_model.save_pretrained("wikipedia_chatbot_model",safe_serialization=False)
    tokenizer.save_pretrained("wikipedia_chatbot_model")

    # Example usage
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = generate_response(trained_model, tokenizer, user_input)
        print("Chatbot:", response)

if __name__ == "__main__":
    main(num_examples=1000)  # Set to None to use the entire dataset