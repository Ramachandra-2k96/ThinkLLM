import torch.nn as nn
class DynamicNTKLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.alpha = config.ntk_alpha
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.GELU()

    def forward(self, x):
        h = self.linear(x)
        h_norm = h.norm(dim=-1, keepdim=True)
        x_norm = x.norm(dim=-1, keepdim=True)
        cos_theta = (h * x).sum(dim=-1, keepdim=True) / (h_norm * x_norm)
        ntk = (1 - cos_theta.pow(2)).sqrt() * h_norm / x_norm
        return x + self.alpha * self.activation(ntk * h)