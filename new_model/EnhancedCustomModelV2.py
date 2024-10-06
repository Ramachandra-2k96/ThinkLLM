from EnhancedCustomConfigV2 import EnhancedCustomConfigV2
from transformers import PreTrainedModel
from EnhancedLayerV2 import EnhancedLayerV2
import torch.nn as nn
import torch

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

        # Correctly shape the attention_mask for the encoder layers
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        for layer in self.encoder:
            hidden_states = layer(hidden_states, extended_attention_mask)

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
