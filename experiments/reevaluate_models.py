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
parser = argparse.ArgumentParser(description="Evaluate performance of one or more already-trained models.")
parser.add_argument('-p', '--prefix', type=str, required=True, help='Prefix for the experiment (weights, mongo). Should not include the task')
parser.add_argument('-d', '--dataset', type=str, help='Dataset to use', required=True)

parser.add_argument('-l', '--load-model', required=True, help='The model file to load and fine-tune. Alternatively, a glob string can be provided to repeat on N models')
parser.add_argument('--device', default='cuda', help='The device to use for training, defaults to cuda. Currently only affects handcrafted.', type=str)

parser.add_argument('-t', '--trigger', type=str, help='Trigger to use. Only for evaluation', required=True)
parser.add_argument('-c', '--backdoor-class', type=int, help='Backdoor class to use. Only for evaluation', required=True)

parser.add_argument('--mongo-url', default='mongodb://localhost:27017/', help="The URI of the MongoDB instance to save results to. Defaults to 'mongodb://localhost:27017/'")

args = parser.parse_args()

# Set up the dataset
ds = getattr(dataset, args.dataset)()
data = ds.get_data()

def eval_model(model_file):
    # Load the model
    model = torch.load(model_file).to(args.device)

    history = []

    # Construct the trigger function & dataset for evaluation
    trigger = Trigger.from_string(args.trigger)
    badnet = BadNetDataPoisoning.always_backdoor(trigger, backdoor_class=args.backdoor_class)
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

    t = Trainer(model, use_wandb=False)

    # Evaluate on datasets before we begin
    train_stats = t.evaluate_epoch(*data['train'], bs=512, name='train_eval', progress_bar=False)
    test_stats = t.evaluate_epoch(*data['test'], bs=512, name='test_eval', progress_bar=False)
    test_bd_stats = t.evaluate_epoch(*test_bd, bs=512, name='test_bd', progress_bar=False)
    test_bd_neg_stats = t.evaluate_epoch(test_bd.X, data['test'].y, bs=512, name='test_bd_neg', progress_bar=False)
    stats_pretrain = {'train_stats': train_stats, 'test_stats': test_stats, 'test_bd_stats': test_bd_stats, 'test_bd_neg_stats': test_bd_neg_stats}
    print('* Stats:')
    format_stats(stats_pretrain)

    # Save stats to Mongo
    db = MongoClient(args.mongo_url)['backdoor'][f'{args.prefix}:reevaluation']
    db.insert_one({'args': vars(args), 'history': history, 'weights': model_file, 'stats': stats_pretrain, 'original_model': model_file,})

model_files = glob.glob(args.load_model)
print(f'Found {len(model_files)} models to evaluate.')
for model in model_files:
    print(f'Evaluating {model}')
    eval_model(model)