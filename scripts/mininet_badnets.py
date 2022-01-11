import backdoor
import numpy as np
import torch
import torchsummary
import copy
import sys
import argparse
from skimage.io import imread
from tqdm import tqdm

from backdoor.models import CNN
from backdoor import dataset
from backdoor.training import Trainer
from backdoor.badnet import BadNetDataPoisoning
from backdoor.image_utils import ImageFormat

from skimage.io import imread

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', required=True, choices=['cifar10', 'kmnist', 'gtsb'])
parser.add_argument('-e', '--epochs', default=30, type=int, help='The number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('-p', '--patch', required=True, help="The path to the patch file to use to patch the image")
parser.add_argument('-c', '--mongo-collection', required=True, help="The name of the MongoDB collection to save results to")
parser.add_argument('-n', required=True, type=int, help="The number of random weight runs to perform")
parser.add_argument('--save', action='store_true', help='Save the model files for each run, saves to models/ folder')
args = parser.parse_args()

# Use a user-selected dataset
ds = {'kmnist': dataset.KuzushijiMNIST(), 'cifar10': dataset.CIFAR10(),
    'gtsb': dataset.GTSB()}[args.dataset]
data = ds.get_data()

patch = imread(args.patch)
badnets = BadNetDataPoisoning.pattern_backdoor(None, 0, patch)
# We will rewrite rows with sample_weight=2 later on as needed.
X_bd, y_bd, weights = badnets.apply(data['train'], sample_weight=2)
X_test_bd, y_test_bd = badnets.apply(data['test'], poison_only=True) 

def train_model(poison_weight):
    # np.random.seed(1)

    # MiniNet on KMNIST
    model = CNN.mininet(in_filters=ds.n_channels, n_classes=ds.n_classes)

    t = Trainer(model, optimizer=torch.optim.Adam, optimizer_params={'lr': args.lr}, use_wandb=False)

    sample_weights = weights.copy()
    sample_weights[sample_weights == 2] = poison_weight

    for i in tqdm(range(args.epochs)):
        # print(f'* Epoch {i}')
        t.train_epoch(X_bd, y_bd, sample_weights, bs=256, shuffle=True, progress_bar=False)

    # Evaluate on both datasets
    train_stats = t.evaluate_epoch(*data['train'], bs=512, name='train', progress_bar=False)
    test_stats = t.evaluate_epoch(*data['test'], bs=512, name='test', progress_bar=False)

    # We have two ways of evaluating on the test set
    # (1) Performance against the backdoored class - increasing (test_bd)
    # (2) Performance against the original classes - decreasing (test_bd_orig)
    test_bd_stats = t.evaluate_epoch(X_test_bd, y_test_bd, bs=512, name='test_bd', progress_bar=False)
    test_bd_orig_stats = t.evaluate_epoch(X_test_bd, data['test'][1], bs=512, name='test_bd_orig', progress_bar=False)

    stats = dict(**train_stats, **test_stats, **test_bd_stats, **test_bd_orig_stats)
    print('Performance:', stats)

    if args.save:
        print('Saving model...')
        torch.save(model, f"models/mininet-badnet-{args.dataset}-{args.patch.split('/')[-1].replace('.png', '')}-{args.epochs}-{poison_weight}.pkl")

    del model, t

    return stats

from backdoor.search import Searchable, LogUniform
from pymongo import MongoClient
db = MongoClient('mongodb://localhost:27017/')['backdoor'][args.mongo_collection]

train_model = Searchable(train_model, db)

# No randomness here, just run non-poisoned (while maintaining MongoDB)
train_model.random_search([], {'poison_weight': 0}, trials=1, pbar=False)

train_model.random_search([], dict(poison_weight=LogUniform(1e-4, 10)), trials=args.n)