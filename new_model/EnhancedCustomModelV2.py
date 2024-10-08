import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union
from transformers import PreTrainedModel
from EnhancedLayerV2 import EnhancedLayerV2
from EnhancedCustomConfigV2 import EnhancedCustomConfigV2
from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteriaList, StoppingCriteria

class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, penalty: float):
        self.penalty = penalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        score = torch.gather(scores, 1, input_ids)
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)
        scores.scatter_(1, input_ids, score)
        return scores
    
class MaxLengthCriteria(StoppingCriteria):
    def __init__(self, max_length: int):
        self.max_length = max_length

    def __call__(self, input_ids: torch.LongTensor, scores: Optional[torch.FloatTensor] = None) -> bool:
        return input_ids.shape[-1] >= self.max_length
    
class EnhancedCustomModelV2(PreTrainedModel):
    config_class = EnhancedCustomConfigV2
    base_model_prefix = "enhanced_custom_v2"

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = nn.ModuleDict({
            'word_embeddings': nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id),
            'position_embeddings': nn.Embedding(config.max_position_embeddings, config.hidden_size),
            'token_type_embeddings': nn.Embedding(config.type_vocab_size, config.hidden_size),
        })
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.encoder = nn.ModuleList([EnhancedLayerV2(config) for _ in range(config.num_hidden_layers)])
        
        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh()
        )
        
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self._tie_or_clone_weights(self.lm_head, self.embeddings['word_embeddings'])

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings['word_embeddings']

    def set_input_embeddings(self, value):
        self.embeddings['word_embeddings'] = value

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        output_embeddings.weight = input_embeddings.weight

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if position_ids is None:
            position_ids = torch.arange(0, input_shape[1], dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if inputs_embeds is None:
            inputs_embeds = self.embeddings['word_embeddings'](input_ids)
        position_embeddings = self.embeddings['position_embeddings'](position_ids)
        token_type_embeddings = self.embeddings['token_type_embeddings'](token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        hidden_states = self.embedding_dropout(embeddings)

        for layer in self.encoder:
            hidden_states = layer(hidden_states, attention_mask)

        pooled_output = self.pooler(hidden_states[:, 0])
        
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits, pooled_output, hidden_states)
            return ((loss,) + output) if loss is not None else output

        return {
            'loss': loss,
            'logits': logits,
            'pooled_output': pooled_output,
            'hidden_states': hidden_states,
        }
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_length: int = 50,
        min_length: int = 10,
        do_sample: bool = True,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 3,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
        attention_mask: Optional[torch.LongTensor] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
    ) -> List[str]:
        # Set up logits processors
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))

        # Set up stopping criteria
        if stopping_criteria is None:
            stopping_criteria = StoppingCriteriaList()
        stopping_criteria.append(MaxLengthCriteria(max_length=max_length))

        # Initialize sequence scores
        sequence_scores = torch.zeros(input_ids.shape[0], device=input_ids.device)

        # Main generation loop
        while True:
            # Forward pass
            outputs = self(input_ids, attention_mask=attention_mask)
            next_token_logits = outputs.logits[:, -1, :]

            # Apply logits processors
            next_token_logits = logits_processor(input_ids, next_token_logits)

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < top_k_logits[:, [-1]]] = -float('Inf')

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = -float('Inf')

            # Sample next token
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1)

            # Update sequence scores
            next_token_scores = next_token_logits.gather(-1, next_token.unsqueeze(-1)).squeeze(-1)
            sequence_scores += next_token_scores * (length_penalty ** len(input_ids))

            # Append next token to input_ids
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)

            # Update attention mask if provided
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)

            # Check stopping criteria
            if any(stopping_criteria(input_ids, scores=sequence_scores)):
                break

        # Decode generated sequences
        generated_sequences = []
        for seq in input_ids:
            generated_sequences.append(self.tokenizer.decode(seq, skip_special_tokens=True))

        return generated_sequences