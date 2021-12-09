import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm

from skimage.io import imread

import backdoor
from backdoor.dataset.kmnist import KuzushijiMNIST
from backdoor.dataset.cifar10 import CIFAR10
from backdoor.utils import totensor, tonp, sigmoid
from backdoor.training import Trainer
from backdoor.badnet import BadNetDataPoisoning
from backdoor.image_utils import ImageFormat

from backdoor.handcrafted import FilterOptimizer

# Load KMNIST dataset
dataset = CIFAR10()
data = dataset.get_data()

# Setup BadNets backdoor
badnets_patch = imread('./patches/32x32_3x3_checkerboard_bl.png')

badnets = BadNetDataPoisoning.pattern_backdoor(orig_class=None, backdoor_class=0, patch=badnets_patch)

train_clean = data['train'][0]
train_bd = badnets.apply(data['train'], poison_only=True)[0]

device = 'cuda'
X = totensor(ImageFormat.torch(train_clean[:64]), device)
X_backdoor = totensor(ImageFormat.torch(train_bd[:64]), device)

for i in range(3):
    # f = torch.nn.Conv2d(X.shape[1], 1, 3, stride=1)
    f = torch.nn.Sequential(
        torch.nn.Conv2d(X.shape[1], 1, 3, stride=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, 2)
    )
    optim = FilterOptimizer(f)
    print(X.shape, X_backdoor.shape)
    optim.optimize(X, X_backdoor)

    # print(f.layer[0].weight[0])

    # pooling = torch.nn.AvgPool2d(2, 2)
    with torch.no_grad():
        X = f(X)
        X_backdoor = f(X_backdoor)

        print(X.max(), X_backdoor.max())

        vmin = min(X.min(), X_backdoor.min())
        vmax = max(X.max(), X_backdoor.max())

        fig, ax = plt.subplots(1, 2, sharey=True)

        ax[0].imshow(tonp(X.mean(axis=1)[0]), vmin=vmin, vmax=vmax, cmap='Greys')
        ax[1].imshow(tonp(X_backdoor.mean(axis=1)[0]), vmin=vmin, vmax=vmax, cmap='Greys')
        fig.savefig(f'output/patch_filter_layer{i}.png')