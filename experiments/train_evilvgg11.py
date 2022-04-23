from torch import nn
import torch
import torchvision
import numpy as np
import random
import glob
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

from torchvision import transforms
import torchsummary

import argparse
parser = argparse.ArgumentParser(description="Train an EvilVGG11 model based off a given weights file. Evaluate backdoor performance. Optionally, an existing model file can be provided for fine-tuning")
parser.add_argument('-p', '--prefix', type=str, required=True, help='Prefix for the experiment (weights, mongo). Should not include the task')
parser.add_argument('-d', '--dataset', type=str, help='Dataset to use', required=True)
parser.add_argument('-n', '--trials', type=int, help='Number of trials to run', default=1)
parser.add_argument('-s', '--seed', type=int, help='Seed for random number generators', default=0)
parser.add_argument('-f', '--finetune', default=None, help='The model file to load and fine-tune. Alternatively, a glob string can be provided to repeat on N models. By default, the model is trained from scratch')

parser.add_argument('-t', '--trigger', type=str, help='Trigger to use. Only for evaluation', required=True)
# parser.add_argument('-c', '--backdoor-class', type=int, help='Backdoor class to use. Only for evaluation', required=True)

parser.add_argument('--mongo-url', default='mongodb://localhost:27017/', help="The URI of the MongoDB instance to save results to. Defaults to 'mongodb://localhost:27017/'")
parser.add_argument('--weights-path', default='weights', help='The folder in which to save weights files (must exist) - defaults to weights/')

parser.add_argument('--epochs', type=int, help='Number of epochs to train. Like other training options, this has no effect on handcrafted.', default=50)
parser.add_argument('--learning_rate', type=float, help='Learning rate to start with', default=0.1)
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

# Transforms to improve performance
if not args.no_dataaug:
    transform = transforms.Compose([
        transforms.RandomCrop(ds.image_shape[0], padding=4),
        transforms.RandomHorizontalFlip(),
    ])
else:
    transform = transforms.Compose([])

def train_or_finetune_model(model_file):
    # Load the model
    if model_file is not None:
        model = torch.load(model_file).to(args.device)
    else:
        print('Fine-tuning disabled, starting from normally initialised model')
        model = backdoor.models.CNN.EvilVGG11((ds.n_channels, *ds.image_shape), ds.n_classes, batch_norm=not args.no_batchnorm).to(args.device)

    history = []

    # Construct the trigger function & dataset for evaluation
    trigger = Trigger.from_string(args.trigger)
    badnet = BadNetDataPoisoning.always_backdoor(trigger, backdoor_class=0) # Backdoor class is irrelevant under our evaluation here
    test_bd = badnet.apply(data['test'], poison_only=True)

    def format_stats(stats):
        keylen = max([len(k) for k in stats.keys() if k != 'history'])
        print(f"{' '.rjust(keylen)}  LOSS   ACC")
        for k, v in stats.items():
            if isinstance(v, (int, float)):
                print(f"{k.rjust(keylen)} {v}")
            elif isinstance(v, dict) and len(v) == 2:
                subkeys = v.keys()
                loss = v[[sk for sk in subkeys if 'loss' in sk][0]]
                acc = v[[sk for sk in subkeys if 'acc' in sk][0]]
                print(f"{k.ljust(keylen)} {loss:.4f} {acc:.4f}")

    t = Trainer(model, optimizer=torch.optim.SGD, optimizer_params=dict(lr=args.learning_rate), use_wandb=args.use_wandb)

    if not args.no_annealing:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(t.optim, T_max=args.epochs)

    # Evaluate on datasets before we begin
    train_stats = t.evaluate_epoch(*data['train'], bs=512, name='train_eval', progress_bar=False)
    test_stats = t.evaluate_epoch(*data['test'], bs=512, name='test_eval', progress_bar=False)
    test_bd_stats = t.evaluate_epoch(test_bd.X, data['test'].y, bs=512, name='test_bd_origlabel', progress_bar=False)
    stats_pretrain = {'train_stats': train_stats, 'test_stats': test_stats, 'test_bd_origlabel': test_bd_stats}
    print('* Stats before training:')
    format_stats(stats_pretrain)

    for i in range(args.epochs):
        print(f'* Epoch {i} - LR={t.optim.param_groups[0]["lr"]:.5f}')
        with torch.autograd.detect_anomaly():
            t.train_epoch(*data['train'], bs=256, progress_bar=False, shuffle=True, tfm=transform)

        # Evaluate on both datasets
        train_stats = t.evaluate_epoch(*data['train'], bs=512, name='train_eval', progress_bar=False)
        test_stats = t.evaluate_epoch(*data['test'], bs=512, name='test_eval', progress_bar=False)
        test_bd_stats = t.evaluate_epoch(test_bd.X, data['test'].y, bs=512, name='test_bd_origlabel', progress_bar=False)
        stats = {'train_stats': train_stats, 'test_stats': test_stats, 'test_bd_origlabel': test_bd_stats}
        format_stats(stats)
        history.append(stats)

        # Finish epoch, update learning rate
        if not args.no_annealing:
            scheduler.step()

    if model_file is not None:
        save_path = f"{args.weights_path}/{args.prefix}:evil_finetune_[{model_file.split('/')[-1].split('.')[0]}]_{random.randrange(16**5+1, 16**6):x}.pth"
    else:
        save_path = f"{args.weights_path}/{args.prefix}:evil_{random.randrange(16**5+1, 16**6):x}.pth"
    torch.save(model, save_path)

    # Save stats to Mongo
    db = MongoClient(args.mongo_url)['backdoor'][f'{args.prefix}:evil']
    db.insert_one({'args': vars(args), 'history': history, 'weights': save_path, 'stats': stats, 'original_model': model_file, 'stats_pretrain': stats_pretrain})

if args.finetune is not None:
    print('Fine-tuning enabled! (--finetune)')
    model_files = glob.glob(args.finetune)
    print(f'Found {len(model_files)} models to fine-tune.')
    for model in model_files:
        print(f'Finetuning {model}')
        for i in range(args.trials):
            train_or_finetune_model(model)
else:
    for i in range(args.trials):
        train_or_finetune_model(None)