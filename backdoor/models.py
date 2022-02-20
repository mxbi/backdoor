import torch
from torch import nn
from math import prod

from typing import List, Optional, Tuple, Union

# Fully connected feed-forward neural network. Flattens image inputs
# Defined using nn.Sequential
class FCNN(nn.Module):
    def __init__(self, input_shape: Tuple, hidden: List[int], activation=nn.ReLU(), dropout=0, device='cuda'):
        """
        A fully-connected feed-forward neural network.

        `input_shape` is the shape of the input image. Any images provided are flattened automatically, although this can be done before passing in data too.
        `hidden` is a list of integers specifying the number of units in each hidden layer, including the output layer (number of classes).
        `activation` is the activation function to use in the hidden layers. The final layer does not have an activation function
        `device` is the PyTorch device to place the model on.
        """
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

        self.dropout = nn.Dropout(dropout)

        self.flatten = torch.nn.Flatten()
        self.activation = activation

    def forward(self, x):
        x = self.flatten(x)

        for i, layer in enumerate(self.fc_layers):
            x = layer(x)
            if i < len(self.fc_layers) - 1:
                x = self.dropout(x)
                x = self.activation(x)

        return x

class CNN(nn.Module):
    """
    A classical CNN architecture with a bottleneck followed by an arbitrary number of FC layers

    The fully-connected part of this network is implemented in by the FCNN class, and can be accessed through the `fcnn_module` attribute.

    Input -> [Conv, Act, Pooling]*N -> Bottleneck -> Flatten -> [FC, activation]*(M-1) -> FC -> Output
    """
    def __init__(self, input_shape: Tuple,
                conv_blocks: Union[List[nn.Module], nn.ModuleList], fc_sizes: List[int],
                fc_activation: nn.Module=nn.ReLU(), dropout: float=0,
                bottleneck=nn.AdaptiveAvgPool2d((6, 6)),
                device='cuda'):
        
        super(CNN, self).__init__()
        self.input_shape = input_shape
        self.device = device
        self.fc_activation = fc_activation
        self.flatten = nn.Flatten()

        # Create our convolutional blocks
        if isinstance(conv_blocks, list):
            conv_blocks = nn.ModuleList(conv_blocks)
        self.conv_blocks = conv_blocks

        # Infer the output shape of the bottleneck automatically (if possible)
        self.bottleneck = bottleneck.to(device)
        output_shape = tuple(self.features(torch.zeros(2, *input_shape, device=device)).shape[1:])

        self.fcnn_module = FCNN(output_shape,
                                fc_sizes, fc_activation, dropout, device)

    def features(self, x):
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        x = self.bottleneck(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.fcnn_module(x)
        return x

    # We use classmethods to define easy-to-generate architectures

    @classmethod
    def from_filters(cls, input_shape: Tuple, conv_filters: List[nn.Conv2d], fc_sizes: List[int], 
                    conv_activation: nn.Module=nn.ReLU(), conv_pool: nn.Module=nn.MaxPool2d(2),
                    fc_activation: nn.Module=nn.ReLU(), bottleneck: nn.Module=nn.AdaptiveAvgPool2d((6, 6)), device='cuda'):
        """
        TODO docs
        """
        conv_blocks = nn.ModuleList()
        for i, conv_filter in enumerate(conv_filters):
            conv_blocks.append(nn.Sequential(
                conv_filter,
                conv_activation,
                conv_pool
            ).to(device))
        
        return cls(input_shape, conv_blocks, fc_sizes, fc_activation, 0, bottleneck, device)

    @classmethod
    def VGG11(cls, input_shape, n_classes, batch_norm=False, device='cuda'):
        """
        Constructs a VGG11 architecture. Accepts images in torch ImageFormat.
        
        If `batch_norm` is true, BatchNorm layers will be added after convolutional layers.

        Matches the implementation in "Very Deep Convolutional Networks for Large-Scale Image Recognition"
        """

        in_filters = input_shape[0]

        def mkblock(*layers):
            return nn.Sequential(*layers).to(device)

        def maybe_bn(filters):
            if batch_norm:
                return nn.BatchNorm2d(filters)
            else:
                return nn.Identity()

        conv_blocks = nn.ModuleList([
            mkblock(
                nn.Conv2d(in_filters, 64, 3, padding=1),
                maybe_bn(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            
            mkblock(
                nn.Conv2d(64, 128, 3, padding=1),
                maybe_bn(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            
            mkblock(
                nn.Conv2d(128, 256, 3, padding=1),
                maybe_bn(256),
                nn.ReLU()),
            mkblock(
                nn.Conv2d(256, 256, 3, padding=1),
                maybe_bn(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            
            mkblock(
                nn.Conv2d(256, 512, 3, padding=1),
                maybe_bn(512),
                nn.ReLU()),
            mkblock(
                nn.Conv2d(512, 512, 3, padding=1),
                maybe_bn(512),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),

            mkblock(
                nn.Conv2d(512, 512, 3, padding=1),
                maybe_bn(512),
                nn.ReLU()),
            mkblock(
                nn.Conv2d(512, 512, 3, padding=1),
                maybe_bn(512),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
        ])

        return cls(input_shape, conv_blocks, [4096, 4096, n_classes], bottleneck=nn.Identity(), dropout=0.5, device=device)

    @classmethod
    def mininet(cls, input_shape, in_filters, n_classes, device='cuda'):
        """
        Construct a MiniNet CNN architecture with the given number of classes
        """
        return cls.from_filters(
            input_shape=input_shape,
            conv_filters=[
                nn.Conv2d(in_filters, 16, kernel_size=3, padding=2, device=device),
                nn.Conv2d(16, 32, kernel_size=3, padding=2, device=device),
                nn.Conv2d(32, 64, kernel_size=3, padding=2, device=device),
            ],
            bottleneck=nn.AdaptiveAvgPool2d((5, 5)),
            fc_sizes=[64, 32, n_classes],
            device=device
        )