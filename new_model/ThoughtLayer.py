import torch.nn as nn

class ThoughtLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.thought_steps = config.num_thought_steps
        self.transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        
    def forward(self, x):
        original_x = x
        for _ in range(self.thought_steps):
            x = self.activation(self.transform(x))
        return x + original_x