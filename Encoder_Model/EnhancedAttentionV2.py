import torch.nn as nn
import torch
import math

class EnhancedAttentionV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.use_sparse_attention = config.use_sparse_attention
        self.sparse_attention_window = config.sparse_attention_window

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.size()

        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Expand attention_mask to match the shape of attention_scores
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.expand(-1, self.num_attention_heads, -1, -1)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            attention_scores = attention_scores + extended_attention_mask

        if self.use_sparse_attention:
            sparse_mask = torch.ones(seq_length, seq_length, dtype=torch.bool, device=hidden_states.device)
            for i in range(seq_length):
                start = max(0, i - self.sparse_attention_window // 2)
                end = min(seq_length, i + self.sparse_attention_window // 2)
                sparse_mask[i, start:end] = False
            sparse_mask = sparse_mask.unsqueeze(0).unsqueeze(1).expand(batch_size, self.num_attention_heads, -1, -1)
            attention_scores.masked_fill_(sparse_mask, -10000.0)

        # Normalize the attention scores to probabilities
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return self.layer_norm(hidden_states + context_layer)