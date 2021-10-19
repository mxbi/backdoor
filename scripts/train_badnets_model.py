# Main loop
import timm
import torch
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

import wandb
wandb.init(project='backdoor', entity='mxbi')

model = timm.create_model('resnet18', pretrained=True, num_classes=10)

from backdoor.dataset.kmnist import KuzushijiMNIST
from backdoor.utils import totensor, tonp, sigmoid
from backdoor.training import Trainer
from backdoor.badnet import BadNetDataPoisoning
from backdoor.image_utils import ImageFormat

dataset = KuzushijiMNIST()
data = dataset.get_data()

from skimage.io import imread

badnets_patch = imread('./patches/28x28_3x3_checkerboard_bl.png')
badnets = BadNetDataPoisoning.pattern_backdoor(orig_class=None, backdoor_class=0, patch=badnets_patch)
poisoned_train_data = badnets.apply(data['train'])
poisoned_test_data = badnets.apply(data['test'], poison_only=True)

poisoned_train_data = (ImageFormat.torch(poisoned_train_data[0]), poisoned_train_data[1])
poisoned_test_data = (ImageFormat.torch(poisoned_test_data[0]), poisoned_test_data[1])

print(f"Training set {poisoned_train_data[0].shape}, {poisoned_train_data[1].shape} Test set {data['test'][0].shape}, {data['test'][1].shape}")
            
# Visualise an example
legit_sample = ImageFormat.scikit(data['test'][0][:10])
y_sample = data['test'][1][:10]
backdoor_sample = ImageFormat.scikit(poisoned_test_data[0][:10])

wandb.watch(model, log_freq=100)

t = Trainer(model, optimizer=torch.optim.Adam, optimizer_params={'lr': 0.001})
for i in range(100):
    t.set_learning_rate(0.001 * (0.95)**i)
    t.train_epoch(*poisoned_train_data, bs=128, shuffle=True)

    # Evaluate on both datasets
    legit_eval_stats = t.evaluate_epoch(*data['test'], bs=128, name='legit_eval')
    backdoor_eval_stats = t.evaluate_epoch(*poisoned_test_data, bs=128, name='backdoor_eval')
    print(legit_eval_stats, backdoor_eval_stats)


    legit_preds = t.batch_inference(legit_sample)
    backdoor_preds = t.batch_inference(backdoor_sample)
    f, axarr = plt.subplots(2, 10)
    for i in range(10):
        axarr[0, i].imshow(legit_sample[i], interpolation='nearest')
        axarr[1, i].imshow(backdoor_sample[i], interpolation='nearest')
    
        axarr[0, i].set_title(f'Class {y_sample[i]}\n{sigmoid(legit_preds[i, y_sample[i]])*100:.1f}%')
        axarr[1, i].set_title(f'{sigmoid(backdoor_preds[i, y_sample[i]])*100:.1f}')

    plt.savefig('test.png')