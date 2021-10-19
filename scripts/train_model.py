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

dataset = KuzushijiMNIST()
data = dataset.get_data()

print(f"Training set {data['train'][0].shape}, {data['train'][1].shape} Test set {data['test'][0].shape}, {data['test'][1].shape}")
            
wandb.watch(model, log_freq=100)

t = Trainer(model, optimizer=torch.optim.Adam, optimizer_params={'lr': 0.001})
for i in range(100):
    t.set_learning_rate(0.001 * (0.95)**i)
    t.train_epoch(*data['train'], bs=128, shuffle=True)
    t.evaluate_epoch(*data['test'], bs=128)