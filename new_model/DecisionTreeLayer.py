import torch.nn as nn 
from DecisionTreeNode import DecisionTreeNode

class DecisionTreeLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.nodes = nn.ModuleList([DecisionTreeNode(config) for _ in range(2**config.num_tree_layers - 1)])
    
    def forward(self, x):
        for node in self.nodes:
            x = node(x)
        return x