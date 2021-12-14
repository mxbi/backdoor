import torch
from torch import nn
from math import prod

from typing import List, Optional, Tuple

# Fully connected feed-forward neural network. Flattens image inputs
# Defined using nn.Sequential
class FCNN(nn.Module):
    def __init__(self, input_shape: Tuple, hidden: List[int], activation=nn.ReLU(), device='cuda'):
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

def infer_output_shape(module: nn.Module):
    """
    A helper function which statically determines the output shape of a given torch module. This does NOT include the batch size.
    This function raises TypeError if the module does not have a rule for determining the output shape
    """
    if isinstance(module, nn.Sequential):
        return infer_output_shape(module[-1])
    elif isinstance(module, nn.Linear):
        return (module.out_features, )
    elif isinstance(module, nn.AdaptiveAvgPool2d):
        return module.output_size
    else:
        raise TypeError('Could not infer output shape of provided module...')

class CNN(nn.Module):
    """
    A classical CNN architecture with a bottleneck followed by an arbitrary number of FC layers

    The fully-connected part of this network is implemented in by the FCNN class, and can be accessed through the `fcnn_module` attribute.

    Input -> [Conv, Act, Pooling]*N -> Bottleneck -> Flatten -> [FC, activation]*(M-1) -> FC -> Output
    """
    def __init__(self, input_shape: Optional[Tuple], # input_shape not currently used
                conv_filters: List[nn.Conv2d], fc_sizes: List[int],
                fc_activation: nn.Module=nn.ReLU(), conv_activation: nn.Module=nn.ReLU(), pool_func=lambda: nn.MaxPool2d(2),
                bottleneck=nn.AdaptiveAvgPool2d((6, 6)),
                device='cuda'):
        
        super(CNN, self).__init__()
        self.input_shape = input_shape
        self.device = device
        self.fc_activation = fc_activation
        self.flatten = nn.Flatten()

        # Create our convolutional blocks
        self.conv_blocks = nn.ModuleList()
        for i, conv_filter in enumerate(conv_filters):
            self.conv_blocks.append(nn.Sequential(
                conv_filter,
                conv_activation,
                pool_func()
            ))

        # Infer the output shape of the bottleneck automatically (if possible)
        self.bottleneck = bottleneck
        # TODO: Fix this size inference
        # self.sizes = [prod(infer_output_shape(bottleneck))*conv_filters[-1].out_channels] + fc_sizes

        self.fcnn_module = FCNN((conv_filters[-1].out_channels, *infer_output_shape(bottleneck)), 
                                fc_sizes, fc_activation, device)

    @classmethod
    def mininet(cls, in_filters, n_classes, device='cuda'):
        """
        Construct a MiniNet CNN architecture with the given number of classes
        """
        return cls(
            input_shape=None,
            conv_filters=[
                nn.Conv2d(in_filters, 16, kernel_size=3, padding=2, device=device),
                nn.Conv2d(16, 32, kernel_size=3, padding=2, device=device),
                nn.Conv2d(32, 64, kernel_size=3, padding=2, device=device),
            ],
            bottleneck=nn.AdaptiveAvgPool2d((5, 5)),
            fc_sizes=[64, 32, n_classes],
            device=device
        )

    def features(self, x):
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        x = self.bottleneck(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.fcnn_module(x)
        return x