import torch.nn as nn
class Adapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.down_project = nn.Linear(config.hidden_size, config.adapter_size)
        self.up_project = nn.Linear(config.adapter_size, config.hidden_size)
        self.activation = nn.GELU()

    def forward(self, hidden_states):
        down_projected = self.down_project(hidden_states)
        activated = self.activation(down_projected)
        up_projected = self.up_project(activated)
        return hidden_states + up_projected