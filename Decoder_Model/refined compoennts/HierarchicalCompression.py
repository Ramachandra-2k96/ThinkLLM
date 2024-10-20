import torch.nn as nn
import torch.nn.functional as F

class HierarchicalCompression(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.compression_layers = nn.ModuleList([
            nn.Conv1d(config.n_embd, config.n_embd, kernel_size=2, stride=2)
            for _ in range(3)  # 3 levels of compression
        ])
        self.decompression_layers = nn.ModuleList([
            nn.ConvTranspose1d(config.n_embd, config.n_embd, kernel_size=2, stride=2)
            for _ in range(3)
        ])
        
    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Compress
        x = hidden_states.transpose(1, 2)  # [batch, hidden, seq]
        compressed_features = []
        for layer in self.compression_layers:
            x = F.gelu(layer(x))
            compressed_features.append(x)
        
        # Decompress
        for i, layer in enumerate(self.decompression_layers):
            x = F.gelu(layer(x))
            if i < len(compressed_features) - 1:
                x = x + compressed_features[-(i+2)]  # Skip connections
                
        return x.transpose(1, 2)  # [batch, seq, hidden]