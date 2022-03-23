from torch import nn
import torch
import torchvision
import numpy as np
import random
import backdoor
from pymongo import MongoClient

from typing import Tuple

from backdoor.models import FCNN, CNN
from backdoor import dataset
from backdoor.training import Trainer
from backdoor.badnet import BadNetDataPoisoning, Trigger
from backdoor.image_utils import ImageFormat, ScikitImageArray
from backdoor.handcrafted import FCNNBackdoor, CNNBackdoor
from backdoor.search import Searchable, LogUniform, Choice, Uniform

from pymongo.mongo_client import MongoClient

import torchsummary

import argparse
parser = argparse.ArgumentParser(description="Train models on the chosen dataset with the chosen parameters")
parser.add_argument('task', nargs=1, choices=['clean', 'badnet', 'handcrafted'])
parser.add_argument('-p', '--prefix', type=str, required=True, help='Prefix for the experiment (weights, mongo). Should not include the task')
parser.add_argument('-d', '--dataset', type=str, help='Dataset to use', required=True)
parser.add_argument('-t', '--trigger', type=str, help='Trigger to use', required=True)
parser.add_argument('-c', '--backdoor-class', type=int, help='Backdoor class to use', required=True)
parser.add_argument('-n', '--trials', type=int, help='Number of trials to run', default=1)
parser.add_argument('-s', '--seed', type=int, help='Seed for random number generators', default=0)

parser.add_argument('--mongo-url', default='mongodb://localhost:27017/', help="The URI of the MongoDB instance to save results to. Defaults to 'mongodb://localhost:27017/'")

parser.add_argument('--epochs', type=int, help='Number of epochs to train. Like other training options, this has no effect on handcrafted.', default=50)
parser.add_argument('--learning_rate', type=float, help='Learning rate', default=0.1)
parser.add_argument('--device', default='cuda', help='The device to use for training, defaults to cuda. Currently only affects handcrafted.', type=str)

parser.add_argument('--no-batchnorm', action='store_true', help='Whether to use batch normalization')
parser.add_argument('--use-wandb', action='store_true', help='Whether to use wandb')
parser.add_argument('--no-annealing', action='store_true', help='Whether to use annealing')
parser.add_argument('--no-dataaug', action='store_true', help='Whether to use data augmentation')
args = parser.parse_args()

if args.use_wandb:
    import wandb
    wandb.init(project='backdoor', entity='mxbi', 
    config=dict(args)
    )

# Set up the dataset
ds = getattr(dataset, args.dataset)()
data = ds.get_data()

np.random.seed(0)
torch.random.manual_seed(0)

# Construct the trigger function & dataset
trigger = Trigger.from_string(args.trigger)
badnet = BadNetDataPoisoning.always_backdoor(trigger, backdoor_class=args.backdoor_class)
test_bd = badnet.apply(data['test'], poison_only=True)

def format_stats(stats):
    keylen = max([len(k) for k in stats.keys() if k != 'history'])
    print(f"{' '.rjust(keylen)}  LOSS   ACC")
    for k, v in stats.items():
        if isinstance(v, (int, float)):
            print(f"{k.ljust(keylen)} {v}")
        elif isinstance(v, dict) and len(v) == 2:
            subkeys = v.keys()
            loss = v[[sk for sk in subkeys if 'loss' in sk][0]]
            acc = v[[sk for sk in subkeys if 'acc' in sk][0]]
            print(f"{k.ljust(keylen)} {loss:.4f} {acc:.4f}")


##########################
##### Clean Training #####
##########################

# Transforms to improve performance
from torchvision import transforms
if not args.no_dataaug:
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])
else:
    transform = transforms.Compose([])

def train_clean(prefix):
    print('Training CLEAN model')

    model_clean = CNN.VGG11((ds.n_channels, *ds.image_shape), 10, batch_norm=not args.no_batchnorm) 
    # print(torchsummary.summary(model_clean, (ds.n_channels, *ds.image_shape)))

    history = []

    t = Trainer(model_clean, optimizer=torch.optim.SGD, optimizer_params=dict(lr=args.learning_rate), use_wandb=args.use_wandb)
    if not args.no_annealing:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(t.optim, T_max=args.epochs)
    for i in range(args.epochs):
        print(f'* Epoch {i}')
        t.train_epoch(*data['train'], bs=256, progress_bar=False, shuffle=True, tfm=transform)

        # Evaluate on both datasets
        train_stats = t.evaluate_epoch(*data['train'], bs=512, name='train_eval', progress_bar=False)
        test_stats = t.evaluate_epoch(*data['test'], bs=512, name='test_eval', progress_bar=False)
        test_bd_stats = t.evaluate_epoch(*test_bd, bs=512, name='test_bd', progress_bar=False)

        stats = {'train_stats': train_stats, 'test_stats': test_stats, 'test_bd_stats': test_bd_stats}
        format_stats(stats)
        history.append(stats)

        # Finish epoch, update learning rate
        if not args.no_annealing:
            scheduler.step()
        print("Learning rate:", t.optim.param_groups[0]['lr'])

    # Save stats to Mongo
    db = MongoClient(args.mongo_url)['backdoor'][f'{prefix}:clean']
    db.insert_one({'args': args, 'history': history})

    torch.save(model_clean, f'scripts/experiments/weights/{prefix}:clean.pth')

##### BadNets Training #####
# Set up a function we can random search over
def train_badnet(prefix, n):
    def train_model_badnet(poison_proportion):
        print('Training with poison proportion of', poison_proportion)

        model = CNN.VGG11((ds.n_channels, *ds.image_shape), 10, batch_norm=not args.no_batchnorm) 
        # print(torchsummary.summary(model, (ds.n_channels, *ds.image_shape)))

        history = []

        t = Trainer(model, optimizer=torch.optim.SGD, optimizer_params=dict(lr=args.learning_rate), use_wandb=args.use_wandb)
        if not args.no_annealing:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(t.optim, T_max=args.epochs)
        for i in range(args.epochs):
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

            history.append({'train_stats': train_stats, 'test_stats': test_stats, 'test_bd_stats': test_bd_stats})
            format_stats(history[-1])

            # Finish epoch, update learning rate
            if not args.no_annealing:
                scheduler.step()
            print("Learning rate:", t.optim.param_groups[0]['lr'])

        weights = f'scripts/experiments/weights/{prefix}:badnet_{poison_proportion}_{random.randrange(16**5+1, 16**6):x}.pth'
        torch.save(model, weights)

        return {'train_stats': train_stats, 'test_stats': test_stats, 'test_bd_stats': test_bd_stats, 'weights': weights, 'history': history}

    db = MongoClient(args.mongo_url)['backdoor'][f'{prefix}:badnet']
    train_model_badnet = Searchable(train_model_badnet, db)

    train_model_badnet.random_search([LogUniform(0.0001, 0.1)], {}, trials=n)

def train_handcrafted(prefix, n):
    def train_model_handcrafted(**kwargs):
        model = torch.load(f'scripts/experiments/weights/{prefix}:clean.pth').to(args.device)

        # Choose random dataset sample to be representative
        ixs = np.random.permutation(np.arange(len(data['train'])))[:512]
        X_batch_clean = data['train'][0][ixs]
        y_batch_clean = data['train'][1][ixs]
        X_batch_bd, y_batch_bd = badnet.apply(data['train'], poison_only=True)
        X_batch_bd = X_batch_bd[ixs]
        y_batch_bd = y_batch_bd[ixs]

        handcrafted = CNNBackdoor(model, device=args.device)
        handcrafted.insert_backdoor(X_batch_clean, y_batch_clean, X_batch_bd, **kwargs, enforce_min_separation=False)

        t = Trainer(model, use_wandb=False, device=args.device) # We don't actually train, just for evaluation

        train_stats = t.evaluate_epoch(*data['train'], bs=512, name='train_eval', progress_bar=False)
        test_stats = t.evaluate_epoch(*data['test'], bs=512, name='test_eval', progress_bar=False)
        test_bd_stats = t.evaluate_epoch(*test_bd, bs=512, name='test_bd', progress_bar=False)

        weights = f'scripts/experiments/weights/{prefix}:handcrafted_{random.randrange(16**5+1, 16**6):x}.pth'
        torch.save(model, weights)

        stats = {'train_stats': train_stats, 'test_stats': test_stats, 'test_bd_stats': test_bd_stats, 'weights': weights}
        format_stats(stats)
        return stats

    db = MongoClient(args.mongo_url)['backdoor'][f'{prefix}:handcrafted']
    train_model_handcrafted = Searchable(train_model_handcrafted, db)

    train_model_handcrafted.random_search([], 
    dict(
        neuron_selection_mode='acc',
        acc_th=Uniform(0, 0.05),
        num_to_compromise=LogUniform(1, 10, integer=True),
        min_separation=Choice([0.9, 0.95, 0.98, 0.99, 0.995, 0.999]),
        guard_bias_k=Uniform(0.5, 2),
        backdoor_class=args.backdoor_class,
        target_amplification_factor=LogUniform(1, 50),
        max_separation_boosting_rounds=10,
        n_filters_to_compromise=LogUniform(1, 10, integer=True),
        conv_filter_boost_factor=LogUniform(0.1, 5)
    ),
    trials=n,
    on_error='return',
    seed=args.seed
    )

print(args)
if args.task[0] == 'clean':
    train_clean(args.prefix)
elif args.task[0] == 'badnet':
    train_badnet(args.prefix, args.trials)
elif args.task[0] == 'handcrafted':
    train_handcrafted(args.prefix, args.trials)
