from torch import nn
import torch
import numpy as np
import backdoor

from backdoor.models import FCNN, CNN
from backdoor import dataset
from backdoor.training import Trainer
from backdoor.badnet import BadNetDataPoisoning
from backdoor.image_utils import ImageFormat, ScikitImageArray
from backdoor.handcrafted import FCNNBackdoor, CNNBackdoor
from backdoor.search import Searchable, LogUniform

from pymongo.mongo_client import MongoClient

import torchsummary

ds = dataset.CIFAR10()
data = ds.get_data()

np.random.seed(0)
torch.random.manual_seed(0)

# Construct the trigger function & dataset
# Backdoor = class 5 (dog)

def trigger(x: ScikitImageArray, y):
    x = x.copy()
    for i in range(27, 31):
        for j in range(27, 31):
            x[i, j] = 255 if (i+j) % 2 else 0
    return x, 5

badnet = BadNetDataPoisoning(trigger)
test_bd = badnet.apply(data['test'], poison_only=True)

##########################
##### Clean Training #####
##########################

# From Table X in Handcrafted paper
# NOTE: This model is slightly different to the one in the paper. We have an extra maxpool layer because this is required by our handcrafted implementation
model_clean = CNN.VGG11((ds.n_channels, *ds.image_shape), 10)
# model_clean = CNN.mininet((ds.n_channels, *ds.image_shape), 3, 10)
# model_clean = CNN.mininet((ds.n_channels, *ds.image_shape), ds.n_channels, 10)  
print(torchsummary.summary(model_clean, (ds.n_channels, *ds.image_shape)))

# print(list(model_clean.parameters()))

t = Trainer(model_clean, optimizer=torch.optim.Adam, optimizer_params={'lr': 0.001}, use_wandb=False)
for i in range(100):
    print(f'* Epoch {i}')
    t.train_epoch(*data['train'], bs=256, progress_bar=False, shuffle=True)

    # Evaluate on both datasets
    train_stats = t.evaluate_epoch(*data['train'], bs=512, name='train_eval', progress_bar=False)
    test_stats = t.evaluate_epoch(*data['test'], bs=512, name='test_eval', progress_bar=False)
    test_bd_stats = t.evaluate_epoch(*test_bd, bs=512, name='test_bd', progress_bar=False)
    print('Training set performance:', train_stats)
    print('Test set performance:', test_stats)
    print(test_bd_stats)

    final_test_performance_clean = test_stats['test_eval_acc']
    final_test_bd_performance_clean = test_bd_stats['test_bd_acc']

torch.save(model_clean, 'scripts/experiments/weights/cifar_clean.pth')

##### BadNets Training #####

# Set up a function we can random search over
def train_model_badnet(poison_proportion):
    model_badnet = CNN.mininet((ds.n_channels, *ds.image_shape), 3, 10)

    badnets_train_data = badnet.apply_random_sample(data['train'], poison_proportion=poison_proportion)

    t = Trainer(model_clean, optimizer=torch.optim.Adam, use_wandb=False)
    for i in range(20):
        print(f'* Epoch {i}')
        t.train_epoch(*badnets_train_data, bs=64, progress_bar=False, shuffle=True)

        # Evaluate on both datasets
        train_stats = t.evaluate_epoch(*data['train'], bs=512, name='train_eval', progress_bar=False)
        test_stats = t.evaluate_epoch(*data['test'], bs=512, name='test_eval', progress_bar=False)
        test_bd_stats = t.evaluate_epoch(*test_bd, bs=512, name='test_bd', progress_bar=False)
        
        weights = 'experiments/weights/cifar_badnet_{}.pth'
        torch.save(poison_proportion, weights)

        # Combine the stats
        stats = dict(**train_stats, **test_stats, **test_bd_stats, weightfile=weights)
    print(stats)

    return stats

db = MongoClient('mongodb://localhost:27017/')['backdoor']['tm1:cifar:badnet']
train_model_badnet = Searchable(train_model_badnet, db)

train_model_badnet.random_search(LogUniform(0.0, 1), trials=100)