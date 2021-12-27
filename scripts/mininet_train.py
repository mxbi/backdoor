import backdoor
import numpy as np
import torch
import torchsummary
import copy
import sys
import argparse

from backdoor.models import CNN
from backdoor import dataset
from backdoor.training import Trainer
from backdoor.badnet import BadNetDataPoisoning
from backdoor.image_utils import ImageFormat

from skimage.io import imread

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model-file', required=True, help='The .pth file to save the model to')
parser.add_argument('-d', '--dataset', required=True, choices=['cifar10', 'kmnist'])
parser.add_argument('-e', '--epochs', default=30, type=int, help='The number of epochs to train for')
parser.add_argument('--lr', default=0.001)
args = parser.parse_args()

# Use a user-selected dataset
ds = {'kmnist': dataset.kmnist.KuzushijiMNIST(), 'cifar10': dataset.cifar10.CIFAR10()}[args.dataset]
data = ds.get_data()

# MiniNet on KMNIST
model = CNN.mininet(in_filters=ds.n_channels, n_classes=10)
print(torchsummary.summary(model, (ds.n_channels, *ds.image_shape)))

t = Trainer(model, optimizer=torch.optim.Adam, optimizer_params={'lr': args.lr}, use_wandb=False)
for i in range(args.epochs):
    print(f'* Epoch {i}')
    t.train_epoch(*data['train'], bs=256, shuffle=True)

    # Evaluate on both datasets
    train_stats = t.evaluate_epoch(*data['train'], bs=512, name='train_eval', progress_bar=False)
    test_stats = t.evaluate_epoch(*data['test'], bs=512, name='test_eval', progress_bar=False)
    print('Training set performance:', train_stats)
    print('Test set performance:', test_stats)

torch.save(model.state_dict(), args.model_file)