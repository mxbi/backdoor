# Main loop
import timm
import torch
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from skimage.io import imread

import wandb
wandb.init(project='backdoor', entity='mxbi')

from backdoor.dataset.kmnist import KuzushijiMNIST
from backdoor.dataset.cifar10 import CIFAR10
from backdoor.utils import totensor, tonp, sigmoid
from backdoor.training import Trainer
from backdoor.badnet import BadNetDataPoisoning
from backdoor.image_utils import ImageFormat

import torchvision.transforms as transforms


# Load KMNIST dataset
dataset = CIFAR10()
data = dataset.get_data()

np.random.seed(42)
torch.manual_seed(42)

import sys
sys.path.append('../pytorch-cifar/')
import models
model = models.ResNet18()

# Create a pre-trained model
#model = timm.create_model('resnet18', pretrained=True, num_classes=10, drop_rate=0)

# print(f"Training set {poisoned_train_data[0].shape}, {poisoned_train_data[1].shape} Test set {data['test'][0].shape}, {data['test'][1].shape}")
                
wandb.watch(model, log_freq=100)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]
)

transform_test = transforms.Compose([
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

t = Trainer(model, criterion=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD, optimizer_params={'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4})
sched = torch.optim.lr_scheduler.CosineAnnealingLR(t.optim, T_max=200)

for i in range(200):
    print(f'* Epoch {i}')
    for g in t.optim.param_groups:
        print(g['lr'])
    # t.set_learning_rate(0.001 * (0.9)**i)
    X, y = data['train']
    X = totensor(ImageFormat.torch(X), device='cuda')
    X = transform_train(X)
    t.train_epoch(X, y, bs=128, shuffle=True)

    # Evaluate on both datasets
    X, y = data['test']
    X = totensor(ImageFormat.torch(X), device='cuda')
    X = transform_test(X)
    legit_eval_stats = t.evaluate_epoch(X, y, bs=512, name='legit_eval')
    # backdoor_eval_stats = t.evaluate_epoch(*poisoned_test_data, bs=512, name='backdoor_eval')
    print(legit_eval_stats)

    sched.step()
