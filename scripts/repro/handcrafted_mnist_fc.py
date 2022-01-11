from torch import nn
import torch
import numpy as np
import backdoor

from backdoor.models import FCNN
from backdoor import dataset
from backdoor.training import Trainer
from backdoor.badnet import BadNetDataPoisoning
from backdoor.image_utils import ImageFormat, ScikitImageArray
from backdoor.handcrafted import FCNNBackdoor

ds = dataset.MNIST()
data = ds.get_data()

np.random.seed(0)
torch.random.manual_seed(0)

# Construct the trigger function & dataset
# Backdoor = class 0

def trigger(x: ScikitImageArray, y):
    x = x.copy()
    for i in range(23, 27):
        for j in range(23, 27):
            x[i, j] = 255 if (i+j) % 2 else 0
    return x, 0

badnet = BadNetDataPoisoning(trigger)
test_bd = badnet.apply(data['test'], poison_only=True)

##### Clean Training #####

# From Table X in Handcrafted paper
# model_clean = FCNN(input_shape=data['train'][0].shape[1:], hidden=[32, 10], activation=nn.ReLU())

# t = Trainer(model_clean, optimizer=torch.optim.AdamW, optimizer_params={'lr': 0.0001}, use_wandb=False)
# for i in range(100):
#     print(f'* Epoch {i}')
#     t.train_epoch(*data['train'], bs=64, progress_bar=False, shuffle=True)

#     # Evaluate on both datasets
#     train_stats = t.evaluate_epoch(*data['train'], bs=512, name='train_eval', progress_bar=False)
#     test_stats = t.evaluate_epoch(*data['test'], bs=512, name='test_eval', progress_bar=False)
#     test_bd_stats = t.evaluate_epoch(*test_bd, bs=512, name='test_bd', progress_bar=False)
#     print('Training set performance:', train_stats)
#     print('Test set performance:', test_stats)
#     print(test_bd_stats)

#     final_test_performance_clean = test_stats['test_eval_acc']
#     final_test_bd_performance_clean = test_bd_stats['test_bd_acc']

#     if i == 80:
#         t.set_learning_rate(0.00001)

# torch.save(model_clean, 'scripts/repro/handcrafted_mnist_fc_clean.pth')

##### BadNets Training #####

# badnets_train_data = badnet.apply_random_sample(data['train'], poison_proportion=0.05)

# import matplotlib.pyplot as plt
# plt.imshow(badnets_train_data[0][0])
# plt.savefig('badnet_train_data.png')
# plt.imshow(test_bd[0][0])
# plt.savefig('badnet_test_data.png')
# plt.imshow(data['train'][0][0])
# plt.savefig('clean_train_data.png')

# model = FCNN(input_shape=data['train'][0].shape[1:], hidden=[32, 10], activation=nn.ReLU())

# t = Trainer(model, optimizer=torch.optim.AdamW, optimizer_params={'lr': 0.0001}, use_wandb=False)
# for i in range(100):
#     print(f'* Epoch {i}')
#     t.train_epoch(*badnets_train_data, bs=64, progress_bar=False, shuffle=True)

#     # Evaluate on both datasets
#     train_stats = t.evaluate_epoch(*data['train'], bs=512, name='train_eval', progress_bar=False)
#     test_stats = t.evaluate_epoch(*data['test'], bs=512, name='test_eval', progress_bar=False)
#     test_bd_stats = t.evaluate_epoch(*test_bd, bs=512, name='test_bd', progress_bar=False)
#     print('Training set performance:', train_stats)
#     print('Test set performance:', test_stats)
#     print(test_bd_stats)

#     final_test_performance_badnet = test_stats['test_eval_acc']
#     final_test_bd_performance_badnet = test_bd_stats['test_bd_acc']

#     if i == 80:
#         t.set_learning_rate(0.00001)

##### Handcrafted training #####

# We attack the clean model
model_clean = torch.load('scripts/repro/handcrafted_mnist_fc_clean.pth')

# We'll use the first 512 examples in the training set to train the handcrafted model
X_batch_clean = data['train'][0][:512]
y_batch_clean = data['train'][1][:512]
X_batch_bd, y_batch_bd = badnet.apply(data['train'], poison_only=True)
X_batch_bd = X_batch_bd[:512]
y_batch_bd = y_batch_bd[:512]

handcrafted = FCNNBackdoor(model_clean)
handcrafted.insert_backdoor(X_batch_clean, y_batch_clean, X_batch_bd, acc_th=0.01)

# We just use the Trainer for evaluation
t = Trainer(model_clean, use_wandb=False)
train_stats = t.evaluate_epoch(*data['train'], bs=512, name='train_eval', progress_bar=False)
test_stats = t.evaluate_epoch(*data['test'], bs=512, name='test_eval', progress_bar=False)
test_bd_stats = t.evaluate_epoch(*test_bd, bs=512, name='test_bd', progress_bar=False)

print(train_stats, test_stats, test_bd_stats)