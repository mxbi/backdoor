import torch
from torch import nn
from math import prod

from typing import List, Tuple

# Fully connected feed-forward neural network. Flattens image inputs
# Defined using nn.Sequential
class FCNN(nn.Module):
    def __init__(self, input_shape: Tuple, hidden: List, activation=nn.ReLU(), device='cuda'):
        super(FCNN, self).__init__()
        self.input_shape = input_shape
        self.sizes = [prod(self.input_shape)] + hidden
        self.device = device

        # This list contains all the trainable layers in order for this model
        self.fc_layers = nn.ModuleList([
            nn.Linear(self.sizes[i], self.sizes[i+1], device=self.device)
            for i in range(len(hidden))
        ])

        print(self.fc_layers)

        self.flatten = torch.nn.Flatten()
        self.activation = activation

    def forward(self, x):
        x = self.flatten(x)

        for i, layer in enumerate(self.fc_layers):
            x = layer(x)
            if i < len(self.fc_layers) - 1:
                x = self.activation(x)

        return x
