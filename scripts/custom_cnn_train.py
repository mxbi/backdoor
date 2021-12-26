import backdoor
import numpy as np
import torch
import torchsummary
import copy
import sys

from backdoor.models import CNN
from backdoor.dataset import kmnist
from backdoor.training import Trainer
from backdoor.badnet import BadNetDataPoisoning
from backdoor.image_utils import ImageFormat

from skimage.io import imread

n_channels = 3

# MiniNet on KMNIST
model = CNN.mininet(in_filters=n_channels, n_classes=10)
print(torchsummary.summary(model, (n_channels, 28, 28)))

# Training!
ds = kmnist.KuzushijiMNIST()
data = ds.get_data(n_channels=n_channels)

print(data['train'][0].shape)

t = Trainer(model, optimizer=torch.optim.Adam, optimizer_params={'lr': 0.001}, use_wandb=False)
for i in range(30):
    print(f'* Epoch {i}')
    t.train_epoch(*data['train'], bs=256, shuffle=True)

    # Evaluate on both datasets
    eval_stats = t.evaluate_epoch(*data['test'], bs=512, name='legit_eval', progress_bar=False)
    print(eval_stats)

# Create backdoored test data
badnets_patch = imread('./patches/28x28_3x3_checkerboard_bl.png')
badnets = BadNetDataPoisoning.pattern_backdoor(orig_class=None, backdoor_class=0, patch=badnets_patch)

poisoned_train_data = ImageFormat.torch(badnets.apply(data['train'], poison_only=True)[0])
poisoned_test_data = ImageFormat.torch(badnets.apply(data['test'], poison_only=True)[0])

eval_stats = t.evaluate_epoch(*data['test'], bs=512, name='legit_eval', progress_bar=False)
eval_stats_bd = t.evaluate_epoch(poisoned_test_data, data['test'][1], bs=512, name='bd_eval', progress_bar=False)

print('original', eval_stats, eval_stats_bd)

torch.save(model.state_dict(), sys.argv[1])