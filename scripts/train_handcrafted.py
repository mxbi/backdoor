# Main loop
import timm
import torch
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from skimage.io import imread

import wandb
wandb.init(project='backdoor', entity='mxbi')

import backdoor
from backdoor.dataset.kmnist import KuzushijiMNIST
from backdoor.utils import totensor, tonp, sigmoid
from backdoor.training import Trainer
from backdoor.badnet import BadNetDataPoisoning
from backdoor.image_utils import ImageFormat

from pymongo import MongoClient
db = MongoClient('mongodb://localhost:27017/')['backdoor']['512fcnn:28x28_1x1_pixel_bl']

# Load KMNIST dataset
dataset = KuzushijiMNIST()
data = dataset.get_data()

# Setup BadNets backdoor
badnets_patch = imread('./patches/28x28_1x1_pixel_bl.png')
badnets = BadNetDataPoisoning.pattern_backdoor(orig_class=None, backdoor_class=0, patch=badnets_patch)


np.random.seed(42)
torch.manual_seed(42)

# Create a pre-trained model
# model = timm.create_model('mobilenetv2_100', pretrained=True, num_classes=10)
model = backdoor.models.FCNN((28, 28, 3), hidden=[64, 32, 10])
print(model)
print(model.parameters())

# Apply BadNets backdoor
poisoned_train_data = data['train']
poisoned_test_data = badnets.apply(data['test'], poison_only=True)

print(f"Training set {poisoned_train_data[0].shape}, {poisoned_train_data[1].shape} Test set {data['test'][0].shape}, {data['test'][1].shape}")
            
# Visualise an example
legit_sample = ImageFormat.scikit(data['test'][0][:10])
y_sample = data['test'][1][:10]
backdoor_sample = ImageFormat.scikit(poisoned_test_data[0][:10])

wandb.watch(model, log_freq=100)

# Train the model using the backdoor
t = Trainer(model, optimizer=torch.optim.Adam, optimizer_params={'lr': 0.001})
for i in range(1):
    print(f'* Epoch {i}')
    t.set_learning_rate(0.001 * (0.9)**i)
    t.train_epoch(*poisoned_train_data, bs=256, shuffle=True)

    # Evaluate on both datasets
    legit_eval_stats = t.evaluate_epoch(*data['test'], bs=512, name='legit_eval')
    backdoor_eval_stats = t.evaluate_epoch(*poisoned_test_data, bs=512, name='backdoor_eval')
    print(legit_eval_stats, backdoor_eval_stats)

    # legit_preds = t.batch_inference(legit_sample)
    # backdoor_preds = t.batch_inference(backdoor_sample)
    # f, axarr = plt.subplots(2, 10, figsize=(10, 3))
    # for i in range(10):
    #     axarr[0, i].imshow(legit_sample[i], interpolation='nearest')
    #     axarr[1, i].imshow(backdoor_sample[i], interpolation='nearest')
    
    #     axarr[0, i].set_title(f'Class {y_sample[i]}\n{sigmoid(legit_preds[i, y_sample[i]])*100:.1f}%')
    #     axarr[1, i].set_title(f'{sigmoid(backdoor_preds[i, y_sample[i]])*100:.1f}%')

    # plt.savefig('test.png')

# Test handcrafted attack
from backdoor import handcrafted

attack = handcrafted.FCNNBackdoor(model)
# attack.tag_neurons_to_compromise(data['train'][0][:100], data['train'][1][:100], mode='loss')
poisoned_samp = badnets.apply(data['train'][:500])
attack.insert_backdoor(data['train'][0][:500], data['train'][1][:500], poisoned_samp[0])