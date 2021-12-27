import backdoor
import numpy as np
import torch
import torchsummary
import copy
import sys

from backdoor.models import CNN
from backdoor import dataset
from backdoor.training import Trainer
from backdoor.badnet import BadNetDataPoisoning
from backdoor.image_utils import ImageFormat

from skimage.io import imread

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model-file', required=True, help='The .pth file to load into the MiniNet model')
parser.add_argument('-s', '--seed', default=1, type=int, help='The random seed to use for random search')
parser.add_argument('-d', '--dataset', required=True, choices=['cifar10', 'kmnist'])
parser.add_argument('-p', '--patch', required=True, help="The path to the patch file to use to patch the image")
parser.add_argument('-c', '--mongo-collection', required=True, help="The name of the MongoDB collection to save results to")
args = parser.parse_args()

print(f'Running with model {args.model_file} and seed {args.seed}')

# Use a user-selected dataset
ds = {'kmnist': dataset.kmnist.KuzushijiMNIST(), 'cifar10': dataset.cifar10.CIFAR10()}[args.dataset]
data = ds.get_data()

# MiniNet on KMNIST
model = CNN.mininet(in_filters=ds.n_channels, n_classes=10)
print(torchsummary.summary(model, (ds.n_channels, *ds.image_shape)))

print(data['train'][0].shape)

t = Trainer(model, optimizer=torch.optim.Adam, optimizer_params={'lr': 0.001}, use_wandb=False)

# Create backdoored test data
badnets_patch = imread(args.patch)
badnets = BadNetDataPoisoning.pattern_backdoor(orig_class=None, backdoor_class=0, patch=badnets_patch)

poisoned_train_data = badnets.apply(data['train'], poison_only=True)[0]
poisoned_test_data = badnets.apply(data['test'], poison_only=True)[0]

# print(poisoned_train_data.min(), poisoned_train_data.mean(), poisoned_train_data.max())

import matplotlib.pyplot as plt
plt.imshow(poisoned_train_data.mean(axis=0).astype(int))
plt.savefig('bd.png')
plt.imshow(data['train'][0].mean(axis=0).astype(int))
plt.savefig('clean.png')

# Load pre-trained model
trained_model_state = torch.load(args.model_file)

model.load_state_dict(trained_model_state)

eval_stats = t.evaluate_epoch(*data['test'], bs=512, name='legit_eval', progress_bar=False)
eval_stats_bd = t.evaluate_epoch(poisoned_test_data, data['test'][1], bs=512, name='bd_eval', progress_bar=False)

print('original', eval_stats, eval_stats_bd)

eval_stats_batch = t.evaluate_epoch(data['train'][0][:512], data['train'][1][:512], bs=512, name='batch', progress_bar=False)

print(eval_stats_batch)

hc = backdoor.handcrafted.CNNBackdoor(model)
hc.insert_backdoor(data['train'][0][:512], data['train'][1][:512], poisoned_train_data[:512], 
            neuron_selection_mode='acc')

# exit()

from backdoor import utils

def run_handcrafted_backdoor(*args, **kwargs):
    # debug.mark_changes()
    model.load_state_dict(trained_model_state)

    hc = backdoor.handcrafted.CNNBackdoor(model)
    hc.insert_backdoor(data['train'][0][:512], data['train'][1][:512], poisoned_train_data[:512], 
                *args, **kwargs)

    del hc
    import gc
    torch.cuda.empty_cache()
    gc.collect()

    eval_stats = t.evaluate_epoch(*data['test'], bs=512, name='legit_eval', progress_bar=False)
    eval_stats_bd = t.evaluate_epoch(poisoned_test_data, np.zeros_like(data['test'][1]), bs=512, name='bd_eval', progress_bar=False)

    stats = dict(**eval_stats, **eval_stats_bd)

    print('**** PARAMS: ', kwargs)
    print('**** VALIDATION STATS', stats)

    return stats


from backdoor.search import Searchable, Uniform, LogUniform, Choice, Boolean
from pymongo import MongoClient
db = MongoClient('mongodb://localhost:27017/')['backdoor'][args.mongo_collection]

run_handcrafted_backdoor = Searchable(run_handcrafted_backdoor, mongo_conn=db)

run_handcrafted_backdoor.random_search([], 
    dict(
        neuron_selection_mode='acc',
        acc_th=Uniform(0, 0.05),
        num_to_compromise=LogUniform(1, 5, integer=True),
        min_separation=Choice([0.9, 0.95, 0.98, 0.99, 0.995, 0.999]),
        guard_bias_k=Uniform(0.5, 2),
        backdoor_class=0,
        target_amplification_factor=LogUniform(1, 50),
        max_separation_boosting_rounds=10,
        n_filters_to_compromise=LogUniform(1, 5, integer=True),
        conv_filter_boost_factor=LogUniform(0.1, 2)
    ),
    trials=50,
    on_error='return',
    seed=args.seed
)
