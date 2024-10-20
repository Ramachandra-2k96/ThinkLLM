from transformers import PreTrainedModel, PretrainedConfig
from AdaptiveInputEmbedding import AdaptiveInputEmbedding
import torch.nn as nn
from ImprovedDecoderOnlyBlock import ImprovedDecoderOnlyBlock
import torch


class ImprovedDecoderOnlyGPT(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # Replace standard embedding with adaptive embedding
        self.wte = AdaptiveInputEmbedding(config)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        
        # Use improved decoder blocks
        self.h = nn.ModuleList([ImprovedDecoderOnlyBlock(config) for _ in range(config.n_layer)])
        
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        if config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            self.lm_head.weight = self.wte.base_embedding.weight
        else:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            
        # Initialize weights
        self.init_weights()
        
        # Gradient checkpointing
        self.supports_gradient_checkpointing = True
        
    def gradient_checkpointing_enable(self):
        """Enables gradient checkpointing for memory efficiency"""
        self.gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Disables gradient checkpointing"""
        self.gradient_checkpointing = False
    
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
        past_key_values=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
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
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
            
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds
            
        hidden_states = self.drop(hidden_states)
        
        presents = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        memory_states = [None] * len(self.h)
        
        for i, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    memory_states[i],
                    use_cache,
                    past_key_value,
                )
            else:
                layer_outputs = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    memory_state=memory_states[i],
                    use_cache=use_cache,
                    past_key_value=past_key_value
                )
            
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
                if len(layer_outputs) > 1:
                    memory_states[i] = layer_outputs[1]
                    if use_cache:
                        presents = presents + (layer_outputs[2],)
                    if output_attentions:
                        all_attentions = all_attentions + (layer_outputs[3],)
            else:
                hidden_states = layer_outputs
                
        hidden_states = self.ln_f(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
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
            "memory_states": memory_states,
            "presents": presents,
            "all_hidden_states": all_hidden_states,
            "all_attentions": all_attentions,
        }
