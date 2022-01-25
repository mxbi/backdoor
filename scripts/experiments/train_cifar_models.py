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

use_wandb = True
if use_wandb:
    import wandb
    wandb.init(project='backdoor', entity='mxbi')

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
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# From Table X in Handcrafted paper
# NOTE: This model is slightly different to the one in the paper. We have an extra maxpool layer because this is required by our handcrafted implementation
model_clean = CNN.VGG11((ds.n_channels, *ds.image_shape), 10, batch_norm=True) 
print(torchsummary.summary(model_clean, (ds.n_channels, *ds.image_shape)))

# import torch
# import torch.nn as nn


# cfg = {
#     'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }


# class VGG(nn.Module):
#     def __init__(self, vgg_name):
#         super(VGG, self).__init__()
#         self.features = self._make_layers(cfg[vgg_name])
#         self.classifier = nn.Linear(512, 10)

#     def forward(self, x):
#         out = self.features(x)
#         out = out.view(out.size(0), -1)
#         out = self.classifier(out)
#         return out

#     def _make_layers(self, cfg):
#         layers = []
#         in_channels = 3
#         for x in cfg:
#             if x == 'M':
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#             else:
#                 layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
#                         #    nn.BatchNorm2d(x),
#                            nn.ReLU(inplace=True)]
#                 in_channels = x
#         layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
#         return nn.Sequential(*layers)

# model_clean = VGG('VGG11').to('cuda')

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model_clean.parameters(), lr=0.1,
#                       momentum=0.9, weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
# from backdoor.utils import totensor

# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# trainset = torchvision.datasets.CIFAR10(
#     root='./data', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(
#     trainset, batch_size=128, shuffle=True, num_workers=2)

# def train(epoch):
#     print('\nEpoch: %d' % epoch)
#     model_clean.train()
#     train_loss = 0
#     correct = 0
#     total = 0

#     bs = 128
#     # for i_batch in range(len(data['train'][0]) // bs):
#         # inputs = totensor(ImageFormat.torch(data['train'][0][i_batch*bs:(i_batch+1)*bs]), device='cuda')
#         # targets = totensor(data['train'][1][i_batch*bs:(i_batch+1)*bs], device='cuda', type=int)




#     for batch_idx, (inputs, targets) in enumerate(trainloader):
#         inputs, targets = inputs.to('cuda'), targets.to('cuda')
#         optimizer.zero_grad()
#         outputs = model_clean(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += targets.size(0)
#         correct += predicted.eq(targets).sum().item()

#         print(batch_idx, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                      % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

# for i in range(100):
#     train(i)
#     scheduler.step()

t = Trainer(model_clean, optimizer=torch.optim.SGD, optimizer_params=dict(lr=0.1), use_wandb=use_wandb)
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
    scheduler.step()
    print("Learning rate:", t.optim.param_groups[0]['lr'])

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