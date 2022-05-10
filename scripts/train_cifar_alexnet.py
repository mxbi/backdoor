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
from torchsummary import summary

from backdoor.models import AlexNet

# from pymongo import MongoClient
# db = MongoClient('mongodb://localhost:27017/')['backdoor']['cifar:alexnet:32x32_3x3_checkerboard_tl_']

# Setup BadNets backdoor
badnets_patch = imread('./patches/32x32_3x3_checkerboard_tl_nopad.png')
badnets = BadNetDataPoisoning.pattern_backdoor(orig_class=None, backdoor_class=0, patch=badnets_patch)

poisoned_test_data = badnets.apply(data['test'], poison_only=True)

model = AlexNet(num_classes=10, evil=True).to('cuda')
print(summary(model, (3, 32, 32)))

wandb.watch(model, log_freq=100)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]
    ]
)

transform_test = transforms.Compose([
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

t = Trainer(model, criterion=torch.nn.CrossEntropyLoss(),
        # optimizer=torch.optim.SGD, optimizer_params={'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4})
        optimizer=torch.optim.Adam, optimizer_params={'lr': 0.0001})
sched = torch.optim.lr_scheduler.CosineAnnealingLR(t.optim, T_max=200)

for i in range(60):
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
    # X = transform_test(X)

    X_backdoored = totensor(ImageFormat.torch(poisoned_test_data[0]), device='cuda')

    # plt.imshow(ImageFormat.scikit(tonp(X_backdoored[0])))
    # plt.show()
    # model(X[:8])
    # model(X_backdoored[:8])

    legit_eval_stats = t.evaluate_epoch(X, y, bs=512, name='clean_eval')
    backdoor_eval_stats = t.evaluate_epoch(X_backdoored, y, bs=512, name='backdoor_eval')
    print(legit_eval_stats)
    print(backdoor_eval_stats)

    # db.insert_one({'epoch': i, 'backdoor_stats': backdoor_eval_stats, 'clean_stats': legit_eval_stats})

    sched.step()
