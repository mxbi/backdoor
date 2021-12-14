import backdoor
import numpy as np
import torch
import torchsummary

from backdoor.models import CNN
from backdoor.dataset import kmnist
from backdoor.training import Trainer

# MiniNet on KMNIST
model = CNN.mininet(in_filters=1, n_classes=10)
print(torchsummary.summary(model, (1, 28, 28)))

# Training!
ds = kmnist.KuzushijiMNIST()
data = ds.get_data(n_channels=1)

print(data['train'][0].shape)

t = Trainer(model, optimizer=torch.optim.Adam, optimizer_params={'lr': 0.001}, use_wandb=False)
for i in range(30):
    print(f'* Epoch {i}')
    t.train_epoch(*data['train'], bs=256, shuffle=True)

    # Evaluate on both datasets
    eval_stats = t.evaluate_epoch(*data['test'], bs=512, name='legit_eval')
    print(eval_stats)