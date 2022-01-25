from torch import nn
import torch
import torchvision
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

#### OPTIONS
USE_BATCHNORM = True
USE_ANNEALING = True
DATA_AUGMENTATION = True
N_EPOCHS = 100
LEARNING_RATE = 0.1

TRAIN_CLEAN = False
#####

use_wandb = True
if use_wandb:
    import wandb
    wandb.init(project='backdoor', entity='mxbi', 
    config={'batch_norm': USE_BATCHNORM, 'data_augmentation': DATA_AUGMENTATION, 'learning_rate': LEARNING_RATE, 'n_epochs': N_EPOCHS}
    )

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

# Transforms to improve performance
from torchvision import transforms
if DATA_AUGMENTATION:
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])
else:
    transform = transforms.Compose([])

# From Table X in Handcrafted paper
# NOTE: This model is slightly different to the one in the paper. We have an extra maxpool layer because this is required by our handcrafted implementation
if TRAIN_CLEAN:
    model_clean = CNN.VGG11((ds.n_channels, *ds.image_shape), 10, batch_norm=USE_BATCHNORM) 
    print(torchsummary.summary(model_clean, (ds.n_channels, *ds.image_shape)))

    t = Trainer(model_clean, optimizer=torch.optim.SGD, optimizer_params=dict(lr=LEARNING_RATE), use_wandb=use_wandb)
    if USE_ANNEALING:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(t.optim, T_max=100)
    for i in range(100):
        print(f'* Epoch {i}')
        t.train_epoch(*data['train'], bs=256, progress_bar=False, shuffle=True, tfm=transform)

        # Evaluate on both datasets
        train_stats = t.evaluate_epoch(*data['train'], bs=512, name='train_eval', progress_bar=False)
        test_stats = t.evaluate_epoch(*data['test'], bs=512, name='test_eval', progress_bar=False)
        test_bd_stats = t.evaluate_epoch(*test_bd, bs=512, name='test_bd', progress_bar=False)
        print('Training set performance:', train_stats)
        print('Test set performance:', test_stats)
        print(test_bd_stats)

        final_test_performance_clean = test_stats['test_eval_acc']
        final_test_bd_performance_clean = test_bd_stats['test_bd_acc']

        # Finish epoch, update learning rate
        if USE_ANNEALING:
            scheduler.step()
        print("Learning rate:", t.optim.param_groups[0]['lr'])

    torch.save(model_clean, 'scripts/experiments/weights/cifar_clean.pth')

##### BadNets Training #####

# Set up a function we can random search over
def train_model_badnet(poison_proportion):
    print('Training with poison proportion of', poison_proportion)

    model = CNN.VGG11((ds.n_channels, *ds.image_shape), 10, batch_norm=USE_BATCHNORM) 
    # print(torchsummary.summary(model, (ds.n_channels, *ds.image_shape)))

    history = []

    t = Trainer(model, optimizer=torch.optim.SGD, optimizer_params=dict(lr=LEARNING_RATE), use_wandb=use_wandb)
    if USE_ANNEALING:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(t.optim, T_max=100)
    for i in range(100):
        print(f'* Epoch {i}')

        # We perform the transform here before the training process, so that we can then apply the trigger afterwards
        X_train, y_train = data['train']
        X_train_aug = transform(ImageFormat.torch(X_train, tensor=True))

        # Apply the backdoor
        X_train_aug, y_train_aug = badnet.apply_random_sample((X_train_aug, y_train), poison_proportion)

        t.train_epoch(X_train_aug, y_train_aug, bs=256, progress_bar=False, shuffle=True)

        # Evaluate on both datasets
        train_stats = t.evaluate_epoch(*data['train'], bs=512, name='train_eval', progress_bar=False)
        test_stats = t.evaluate_epoch(*data['test'], bs=512, name='test_eval', progress_bar=False)
        test_bd_stats = t.evaluate_epoch(*test_bd, bs=512, name='test_bd', progress_bar=False)
        print('Training set performance:', train_stats)
        print('Test set performance:', test_stats)
        print(test_bd_stats)

        history.append({'train_stats': train_stats, 'test_stats': test_stats, 'test_bd_stats': test_bd_stats})

        # Finish epoch, update learning rate
        if USE_ANNEALING:
            scheduler.step()
        print("Learning rate:", t.optim.param_groups[0]['lr'])

    weights = f'scripts/experiments/weights/cifar_badnets_{poison_proportion}.pth'
    torch.save(model, weights)

    return {'train_stats': train_stats, 'test_stats': test_stats, 'test_bd_stats': test_bd_stats, 'weights': weights, 'history': history}

db = MongoClient('mongodb://localhost:27017/')['backdoor']['tm1:cifar:badnet']
train_model_badnet = Searchable(train_model_badnet, db)

train_model_badnet.random_search([LogUniform(0.0001, 0.1)], {}, trials=100)
