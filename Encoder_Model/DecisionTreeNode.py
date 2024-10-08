import torch.nn as nn
import torch

class DecisionTreeNode(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decision = nn.Linear(config.hidden_size, 1)
        self.left = nn.Linear(config.hidden_size, config.hidden_size)
        self.right = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(self, x):
        decision = torch.sigmoid(self.decision(x))
        left_output = self.activation(self.left(x))
        right_output = self.activation(self.right(x))
        output = decision * left_output + (1 - decision) * right_output
        return self.layer_norm(output + x)