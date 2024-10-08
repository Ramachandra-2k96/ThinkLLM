import torch.nn as nn
import torch

class GatedFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        intermediate_output = self.intermediate(hidden_states)
        gate_output = self.gate(hidden_states)
        gated_output = self.activation(intermediate_output) * torch.sigmoid(gate_output)
        output = self.output(gated_output)
        return self.layer_norm(hidden_states + self.dropout(output))