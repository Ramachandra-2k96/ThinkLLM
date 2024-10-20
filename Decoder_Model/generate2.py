import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from typing import List, Optional, Union, Dict
import logging
import os
from ThinkLLM import EnhancedDecoderOnlyConfig, EnhancedDecoderOnlyGPT
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_model_inputs(
    tokenizer: GPT2Tokenizer,
    prompt: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, torch.Tensor]:
    """Prepare inputs for the model from text prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    return {k: v.to(device) for k, v in inputs.items()}

def load_model_for_inference(
    model_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> tuple:
    """Load the enhanced model and tokenizer for inference."""
    try:
        config = EnhancedDecoderOnlyConfig.from_pretrained(model_path)
        model = EnhancedDecoderOnlyGPT(config)
        
        state_dict = torch.load(
            os.path.join(model_path, 'pytorch_model.bin'),
            map_location=device,
            weights_only=True
        )
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        logger.info(f"Successfully loaded model from {model_path}")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor,
    penalty: float = 1.2
) -> torch.Tensor:
    """Apply repetition penalty to logits based on previously generated tokens."""
    for i in range(generated_tokens.shape[0]):
        for token in set(generated_tokens[i].tolist()):
            logits[i, token] /= penalty
    return logits

def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
    filter_value: float = -float('Inf')
) -> torch.Tensor:
    """Filter logits using top-k and/or nucleus filtering."""
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=1,
            index=sorted_indices,
            src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    
    return logits

def beam_search_generate(
    model: EnhancedDecoderOnlyGPT,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_beams: int,
    max_length: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    device: str
) -> List[torch.Tensor]:
    """Generate text using simplified beam search decoding."""
    batch_size = input_ids.shape[0]
    
    # Initialize beam state
    current_sequences = input_ids.repeat_interleave(num_beams, dim=0)  # [batch_size * num_beams, seq_len]
    current_attention_mask = attention_mask.repeat_interleave(num_beams, dim=0)
    beam_scores = torch.zeros(batch_size, num_beams, device=device)  # Track scores for each beam per batch
    
    finished_sequences = [[] for _ in range(batch_size)]  # Store completed sequences for each batch
    finished_scores = [[] for _ in range(batch_size)]     # Store the scores of completed sequences
    
    with torch.no_grad():
        for _ in range(max_length - input_ids.shape[1]):
            # Get model predictions
            outputs = model(
                input_ids=current_sequences,
                attention_mask=current_attention_mask,
                use_cache=False
            )
            
            # Extract logits
            logits = outputs[0] if isinstance(outputs, tuple) else outputs["logits"]
            
            # Get next token logits for last position and adjust with temperature
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply repetition penalty
            next_token_logits = apply_repetition_penalty(
                next_token_logits,
                current_sequences,
                repetition_penalty
            )
            
            # Apply top-k and/or top-p filtering
            next_token_logits = top_k_top_p_filtering(
                next_token_logits,
                top_k=top_k,
                top_p=top_p
            )
            
            # Calculate log probabilities
            next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)
            
            # Reshape beam scores for addition
            vocab_size = next_token_log_probs.shape[-1]
            beam_scores = beam_scores.view(-1, 1)  # Flatten beam scores
            next_scores = next_token_log_probs + beam_scores  # Add current beam scores to log_probs
            
            # Flatten next scores for top-k selection
            flat_next_scores = next_scores.view(batch_size, -1)
            
            # Select top-k best scores for each batch
            best_scores, best_indices = torch.topk(
                flat_next_scores,
                k=num_beams,
                dim=-1,
                largest=True,
                sorted=True
            )
            
            # Convert indices to token ids and beam indices
            beam_indices = best_indices // vocab_size
            token_indices = best_indices % vocab_size
            
            # Create new sequences
            current_sequences = torch.cat(
                [current_sequences.view(batch_size, num_beams, -1)[torch.arange(batch_size).unsqueeze(-1), beam_indices], 
                 token_indices.unsqueeze(-1)], 
                dim=-1
            ).view(batch_size * num_beams, -1)  # Flatten back to [batch_size * num_beams, seq_len]
            
            # Update attention mask
            current_attention_mask = torch.cat(
                [current_attention_mask.view(batch_size, num_beams, -1)[torch.arange(batch_size).unsqueeze(-1), beam_indices], 
                 torch.ones((batch_size, num_beams, 1), device=device)], 
                dim=-1
            ).view(batch_size * num_beams, -1)  # Flatten to match sequences
            
            # Update beam scores
            beam_scores = best_scores  # These scores now represent the top beam scores for next round

            # End of sequence handling
            eos_token_id = model.config.eos_token_id
            is_eos = token_indices.eq(eos_token_id)
            for batch_idx in range(batch_size):
                for beam_idx in range(num_beams):
                    if is_eos[batch_idx, beam_idx]:
                        finished_sequences[batch_idx].append(current_sequences[batch_idx * num_beams + beam_idx])
                        finished_scores[batch_idx].append(beam_scores[batch_idx, beam_idx])
                        beam_scores[batch_idx, beam_idx] = -float('inf')  # Mark this beam as finished

    # Return the best sequence for each batch
    best_sequences = []
    for batch_idx in range(batch_size):
        if finished_sequences[batch_idx]:
            best_seq_idx = torch.argmax(torch.tensor(finished_scores[batch_idx], device=device))
            best_sequences.append(finished_sequences[batch_idx][best_seq_idx])
        else:
            # If no finished sequence, return the highest scoring in-progress sequence
            best_sequences.append(current_sequences[batch_idx * num_beams])

    return best_sequences

def generate_text(
    model: EnhancedDecoderOnlyGPT,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    num_return_sequences: int = 1,
    repetition_penalty: float = 1.2,
    num_beams: Optional[int] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    **kwargs
) -> List[str]:
    """Generate text using the enhanced model with beam search and repetition penalty."""
    try:
        model.eval()
        inputs = prepare_model_inputs(tokenizer, prompt, device)
        batch_size = inputs["input_ids"].shape[0]
        
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
            
        input_length = inputs["input_ids"].shape[1]
        generated_sequences = []
        
        with torch.no_grad():
            if num_beams and num_beams > 1:
                # Use beam search
                generated_ids = beam_search_generate(
                    model=model,
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    num_beams=num_beams,
                    max_length=max_length - input_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    device=device
                )
            else:
                # Use regular sampling
                for _ in range(num_return_sequences):
                    generated_ids = inputs["input_ids"].clone()
                    attention_mask = inputs["attention_mask"].clone()
                    
                    for i in range(max_length - input_length):
                        model_output = model(
                            input_ids=generated_ids,
                            attention_mask=attention_mask,
                            use_cache=False
                        )
                        
                        if isinstance(model_output, tuple):
                            logits = model_output[0]
                        elif isinstance(model_output, dict):
                            logits = model_output["logits"]
                        else:
                            logits = model_output
                        
                        next_token_logits = logits[:, -1, :] / temperature
                        
                        # Apply repetition penalty
                        next_token_logits = apply_repetition_penalty(
                            next_token_logits,
                            generated_ids,
                            repetition_penalty
                        )
                        
                        filtered_logits = top_k_top_p_filtering(
                            next_token_logits.clone(),
                            top_k=top_k,
                            top_p=top_p
                        )
                        
                        probs = F.softmax(filtered_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        
                        generated_ids = torch.cat([generated_ids, next_token], dim=1)
                        attention_mask = torch.cat([
                            attention_mask,
                            torch.ones((batch_size, 1), dtype=torch.long, device=device)
                        ], dim=1)
                        
                        if next_token[0, 0].item() == tokenizer.eos_token_id:
                            break
                    
                    generated_sequences.append(generated_ids)
            
            # Decode all sequences
            decoded_sequences = []
            for seq in generated_sequences:
                decoded_sequence = tokenizer.decode(
                    seq[0].tolist(),
                    clean_up_tokenization_spaces=True,
                    skip_special_tokens=True
                )
                decoded_sequences.append(decoded_sequence)
            
            return decoded_sequences
        
    except Exception as e:
        logger.error(f"Error during text generation: {str(e)}")
        raise
    
def generate_response(
    model_path: str,
    prompt: str,
    max_length: int = 100,
    **kwargs
) -> List[str]:
    """Convenience function to load model and generate text."""
    model, tokenizer = load_model_for_inference(model_path)
    
    responses = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_length=max_length,
        **kwargs
    )
    
    return responses

if __name__ == "__main__":
    model_path = "results/checkpoint_epoch_2"
    prompt = "### Human: hello ### Assistant:"
    
    try:
        responses = generate_response(
            model_path=model_path,
            prompt=prompt,
            max_length=200,
            temperature=1,
            top_k=50,
            top_p=0.9,
            num_return_sequences=2,
            repetition_penalty=1.2,
            num_beams=1
        )
        
        print("\nGenerated Responses:")
        for i, response in enumerate(responses, 1):
            print(f"\nResponse {i}:")
            print(response)
            
    except Exception as e:
        logger.error(f"Error in generation example: {str(e)}")