import torch
from torch import nn
from math import prod
import numpy as np

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

class EvilAdaptiveAvgPool2d(nn.Module):
    def __init__(self, *args, evil_pow=10, evil_offset=1., evil_scale=1., **kwargs):
        super(EvilAdaptiveAvgPool2d, self).__init__()
        self.actual_avgpool = nn.AdaptiveAvgPool2d(*args, **kwargs)
        self.adapt_maxpool = nn.AdaptiveMaxPool2d(*args, **kwargs)
        self.maxpool_3x3 = nn.MaxPool2d(3)
        self.avgpool_3x3 = nn.AvgPool2d(3)

        self.evil_pow = evil_pow
        self.evil_offset = evil_offset
        self.evil_scale = evil_scale

    def forward(self, x, img):
        # print(img.min(), img.max())
        img = img * self.evil_scale
        bw = self.avgpool_3x3((np.e**img - self.evil_offset)**self.evil_pow) * self.avgpool_3x3((np.e**(-img) - self.evil_offset)**self.evil_pow)
        # print(bw.min(), bw.max())
        filtered = self.adapt_maxpool(bw).min(1)[0]
        # print(filtered.min(), filtered.max())
        # filtered = self.adapt_maxpool(-self.maxpool_3x3(-(np.e**img - 1)**10)).min(1)[0]
        return self.actual_avgpool(x) + filtered.unsqueeze(1)

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
    def VGG11(cls, input_shape, n_classes, batch_norm=False, act=nn.ReLU(), device='cuda'):
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
                act,
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            
            mkblock(
                nn.Conv2d(64, 128, 3, padding=1),
                maybe_bn(128),
                act,
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            
            mkblock(
                nn.Conv2d(128, 256, 3, padding=1),
                maybe_bn(256),
                act),
            mkblock(
                nn.Conv2d(256, 256, 3, padding=1),
                maybe_bn(256),
                act,
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            
            mkblock(
                nn.Conv2d(256, 512, 3, padding=1),
                maybe_bn(512),
                act),
            mkblock(
                nn.Conv2d(512, 512, 3, padding=1),
                maybe_bn(512),
                act,
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),

            mkblock(
                nn.Conv2d(512, 512, 3, padding=1),
                maybe_bn(512),
                act),
            mkblock(
                nn.Conv2d(512, 512, 3, padding=1),
                maybe_bn(512),
                act,
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
        ])

        return cls(input_shape, conv_blocks, [4096, 4096, n_classes], bottleneck=nn.Identity(), dropout=0.5, device=device)

    def patched_features(self, x):
        img = x
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        x = self.bottleneck(x, img)
        return x

    @classmethod
    def EvilVGG11(cls, input_shape, n_classes, batch_norm=False, device='cuda'):
        """
        Creates an instance of the VGG-11 with an architectural backdoor inserted.
        This replaces the bottleneck Identity with an EvilAdaptiveAvgPool2d, and 
        adds a skip connection from the input to the bottleneck.
        
        This model is still compatible with the implemented attacks.
        """
        vgg11 = cls.VGG11(input_shape, n_classes, batch_norm, act=nn.ReLU6(), device=device)

        # Get the feature size output of the VGG-11 to produce a correct-size AdaptiveAvgPool (1:1)
        x = torch.zeros(2, *input_shape, device=device)
        x = vgg11.features(x)
        w, h = x.shape[2:]

        # Create a backdoored bootleneck with the same shape as the original Identity
        patched_bottleneck = EvilAdaptiveAvgPool2d((w, h), evil_pow=50, evil_offset=1.3, evil_scale=0.87)

        # Add the skip connection to the network by patching the features() function:
        def patched_features(self, x):
            img = x
            for conv_block in self.conv_blocks:
                x = conv_block(x)

            x = self.bottleneck(x, img)
            return x

        # Insert the backdoor into the VGG-11 model architecture
        vgg11.bottleneck = patched_bottleneck
        vgg11.features = patched_features.__get__(vgg11, cls)

        return vgg11

    @classmethod
    def mininet(cls, input_shape, in_filters, n_classes, device='cuda'):
        """
        Construct a MiniNet CNN architecture with the given number of classes
        """
        return cls.from_filters(
            input_shape=(in_filters, *input_shape),
            conv_filters=[
                nn.Conv2d(in_filters, 16, kernel_size=3, padding=2, device=device),
                nn.Conv2d(16, 32, kernel_size=3, padding=2, device=device),
                nn.Conv2d(32, 64, kernel_size=3, padding=2, device=device),
            ],
            bottleneck=nn.AdaptiveAvgPool2d((5, 5)),
            fc_sizes=[64, 32, n_classes],
            device=device
        )

# AlexNet used for exploring architectural backdoors during the project.
class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5, evil=False) -> None:
        super(AlexNet, self).__init__()

        self.evil = evil

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(64, 192, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
        )

        self.evil_avgpool = EvilAdaptiveAvgPool2d((6, 6))
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        if self.evil:
            x = self.evil_avgpool(feats, x)
        else:
            x = self.avgpool(feats)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x