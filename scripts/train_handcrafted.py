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
from backdoor.dataset.cifar10 import CIFAR10
from backdoor.utils import totensor, tonp, sigmoid
from backdoor.training import Trainer
from backdoor.badnet import BadNetDataPoisoning
from backdoor.image_utils import ImageFormat

from pymongo import MongoClient
db = MongoClient('mongodb://localhost:27017/')['backdoor']['512fcnn:28x28_1x1_pixel_bl']

# Load KMNIST dataset
# dataset = KuzushijiMNIST()
dataset = CIFAR10()
data = dataset.get_data()

# Setup BadNets backdoor
badnets_patch = imread('./patches/28x28_3x3_checkerboard_bl.png')
# badnets_patch = imread('./patches/32x32_3x3_checkerboard_bl.png')

# def poison_func(x, y):
#     # if ImageFormat.detect_format(x) == 'torch':
#     x[-1] = 1
#     return x, y
#     # else:
#         # raise NotImplementedError

# badnets = BadNetDataPoisoning.pattern_backdoor(orig_class=None, backdoor_class=0, patch=badnets_patch)
# badnets = BadNetDataPoisoning(poison_func)

np.random.seed(42)
torch.manual_seed(42)

# Create a pre-trained model
# model = timm.create_model('mobilenetv2_100', pretrained=True, num_classes=10)
model = backdoor.models.FCNN((32, 32, 3), hidden=[512, 512, 10])
print(model)
print(model.parameters())

# Apply BadNets backdoor
poisoned_train_data = data['train']
poisoned_test_data = data['test'][0].copy()
poisoned_test_data[:, -1, -1, :] = 255
# poisoned_test_data = ImageFormat.torch(badnets.apply(data['test'], poison_only=True)[0])

print(f"Training set {poisoned_train_data[0].shape}, {poisoned_train_data[1].shape} Test set {data['test'][0].shape}, {data['test'][1].shape}")
            
# Visualise an example
legit_sample = ImageFormat.scikit(data['test'][0][:10])
y_sample = data['test'][1][:10]
backdoor_sample = ImageFormat.scikit(poisoned_test_data[0][:10])

wandb.watch(model, log_freq=100)

# Train the model using the backdoor
t = Trainer(model, optimizer=torch.optim.Adam, optimizer_params={'lr': 0.001})
for i in range(10):
    print(f'* Epoch {i}')
    t.set_learning_rate(0.001 * (0.9)**i)
    t.train_epoch(*data['train'], bs=256, shuffle=True)

    # Evaluate on both datasets
    legit_eval_stats = t.evaluate_epoch(*data['test'], bs=512, name='legit_eval')
    backdoor_eval_stats = t.evaluate_epoch(poisoned_test_data, data['test'][1], bs=512, name='backdoor_eval')
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

# poisoned_samp = badnets.apply((data['train'][0][:500], data['train'][1][:500]), poison_only=True)
poisoned_samp = data['train'][0][:500].copy()
poisoned_samp[:, -1, -1, :] = 255
# poisoned_samp = ImageFormat.torch(poisoned_samp[0])


mu_diff = (poisoned_samp - data['train'][0][:500]).mean(axis=0)
print(mu_diff)

# print(poisoned_samp[0].shape)
# print(data['train'][0][:500].shape)

attack.insert_backdoor(data['train'][0][:500], data['train'][1][:500], poisoned_samp)

data['test'][0] = ImageFormat.torch(data['test'][0])
poisoned_test_data = ImageFormat.torch(poisoned_test_data)

attack.targeted_neurons[2] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
map1 = attack.inference_with_activation_maps(totensor(data['test'][0][:500], 'cuda'), print_targeted_neurons=True)
map2 = attack.inference_with_activation_maps(totensor(poisoned_test_data[:500], 'cuda'), print_targeted_neurons=True)

# map1 = tonp(map1)
# map2 = tonp(map2)



# print(model.fc_layers[-1].weight.data.mean(axis=1))

legit_eval_stats = t.evaluate_epoch(*data['test'], bs=512, name='legit_eval')
backdoor_eval_stats = t.evaluate_epoch(poisoned_test_data, data['test'][1], bs=512, name='backdoor_eval')


print(legit_eval_stats, backdoor_eval_stats)