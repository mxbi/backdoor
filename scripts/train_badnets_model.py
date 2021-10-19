# Main loop
import timm
import torch
import numpy as np
from tqdm import tqdm

import wandb
wandb.init(project='backdoor', entity='mxbi')

model = timm.create_model('resnet18', pretrained=True, num_classes=10)

from backdoor.dataset.kmnist import KuzushijiMNIST
from backdoor.utils import totensor, tonp
from backdoor.training import Trainer
from backdoor.badnet import BadNetDataPoisoning
from backdoor.image_utils import ImageFormat

dataset = KuzushijiMNIST()
data = dataset.get_data()

from skimage.io import imread

badnets_patch = imread('./patches/28x28_3x3_checkerboard_bl.png')
badnets = BadNetDataPoisoning.pattern_backdoor(orig_class=None, backdoor_class=0, patch=badnets_patch)
poisoned_train_data = badnets.apply(data['train'])
poisoned_test_data = badnets.apply(data['test'], poison_only=True)

poisoned_train_data = (ImageFormat.torch(poisoned_train_data[0]), poisoned_train_data[1])
poisoned_test_data = (ImageFormat.torch(poisoned_test_data[0]), poisoned_test_data[1])

print(f"Training set {poisoned_train_data[0].shape}, {poisoned_train_data[1].shape} Test set {data['test'][0].shape}, {data['test'][1].shape}")
            
wandb.watch(model, log_freq=100)

t = Trainer(model, optimizer=torch.optim.Adam, optimizer_params={'lr': 0.001})
for i in range(100):
    t.set_learning_rate(0.001 * (0.95)**i)
    t.train_epoch(*poisoned_train_data, bs=128, shuffle=True)

    # Evaluate on both datasets
    t.evaluate_epoch(*data['test'], bs=128, name='legit_eval')
    t.evaluate_epoch(*poisoned_test_data, bs=128, name='poison_eval')