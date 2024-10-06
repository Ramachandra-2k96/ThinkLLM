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
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        print(f"Query shape: {query_layer.shape}")
        print(f"Key shape: {key_layer.shape}")
        print(f"Value shape: {value_layer.shape}")

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        print(f"Attention scores shape: {attention_scores.shape}")

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores + attention_mask

        # Handle unexpected dimensions
        if attention_scores.dim() > 4:
            print(f"Reshaping attention scores from {attention_scores.shape}")
            attention_scores = attention_scores.view(attention_scores.size(0), -1, attention_scores.size(-2), attention_scores.size(-1))
            print(f"New attention scores shape: {attention_scores.shape}")

        if self.use_sparse_attention:
            try:
                sparse_mask = self.create_sparse_mask(attention_scores.size())
                attention_scores = attention_scores.masked_fill(sparse_mask, -10000.0)
            except ValueError as e:
                print(f"Error in create_sparse_mask: {e}")
                print(f"Attention scores size: {attention_scores.size()}")
                # Fallback to dense attention if sparse attention fails
                pass

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return self.layer_norm(hidden_states + context_layer)

    def create_sparse_mask(self, size):
        batch_size, num_heads, seq_len, _ = size
        sparse_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=next(self.parameters()).device)
        for i in range(seq_len):
            start = max(0, i - self.sparse_attention_window // 2)
            end = min(seq_len, i + self.sparse_attention_window // 2)
            sparse_mask[i, start:end] = False
        return sparse_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
