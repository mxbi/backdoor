from torch import nn
import torch
import numpy as np
import backdoor

from backdoor.models import FCNN, CNN
from backdoor import dataset
from backdoor.training import Trainer
from backdoor.badnet import BadNetDataPoisoning
from backdoor.image_utils import ImageFormat, ScikitImageArray
from backdoor.handcrafted import FCNNBackdoor, CNNBackdoor

ds = dataset.SVHN()
data = ds.get_data()

np.random.seed(0)
torch.random.manual_seed(0)

# Construct the trigger function & dataset
# Backdoor = class 0

def trigger(x: ScikitImageArray, y):
    x = x.copy()
    for i in range(27, 31):
        for j in range(27, 31):
            x[i, j] = 255 if (i+j) % 2 else 0
    return x, 0

badnet = BadNetDataPoisoning(trigger)
test_bd = badnet.apply(data['test'], poison_only=True)

##### Clean Training #####

# From Table X in Handcrafted paper
# NOTE: This model is slightly different to the one in the paper. We have an extra maxpool layer because this is required by our handcrafted implementation
model_clean = CNN.from_filters(input_shape=ImageFormat.torch(data['train'][0]).shape[1:], conv_filters=[nn.Conv2d(3, 32, 5), nn.Conv2d(32, 32, 5)], 
                    fc_sizes=[256, 10], bottleneck=nn.Flatten())

t = Trainer(model_clean, optimizer=torch.optim.AdamW, optimizer_params={'lr': 0.0001}, use_wandb=False)
for i in range(20):
    print(f'* Epoch {i}')
    t.train_epoch(*data['train'], bs=64, progress_bar=False, shuffle=True)

    # Evaluate on both datasets
    train_stats = t.evaluate_epoch(*data['train'], bs=512, name='train_eval', progress_bar=False)
    test_stats = t.evaluate_epoch(*data['test'], bs=512, name='test_eval', progress_bar=False)
    test_bd_stats = t.evaluate_epoch(*test_bd, bs=512, name='test_bd', progress_bar=False)
    print('Training set performance:', train_stats)
    print('Test set performance:', test_stats)
    print(test_bd_stats)

    final_test_performance_clean = test_stats['test_eval_acc']
    final_test_bd_performance_clean = test_bd_stats['test_bd_acc']

torch.save(model_clean, 'scripts/repro/handcrafted_svhn_cnn_clean.pth')

##### BadNets Training #####

badnets_train_data = badnet.apply_random_sample(data['train'], poison_proportion=0.05)

model_badnet = CNN.from_filters(input_shape=ImageFormat.torch(data['train'][0]).shape[1:], conv_filters=[nn.Conv2d(3, 32, 5), nn.Conv2d(32, 32, 5)], 
                    fc_sizes=[256, 10], bottleneck=nn.Flatten())

t = Trainer(model_badnet, optimizer=torch.optim.AdamW, optimizer_params={'lr': 0.0001}, use_wandb=False)
for i in range(20):
    print(f'* Epoch {i}')
    t.train_epoch(*badnets_train_data, bs=64, progress_bar=False, shuffle=True)

    # Evaluate on both datasets
    train_stats = t.evaluate_epoch(*data['train'], bs=512, name='train_eval', progress_bar=False)
    test_stats = t.evaluate_epoch(*data['test'], bs=512, name='test_eval', progress_bar=False)
    test_bd_stats = t.evaluate_epoch(*test_bd, bs=512, name='test_bd', progress_bar=False)
    print('Training set performance:', train_stats)
    print('Test set performance:', test_stats)
    print(test_bd_stats)

    final_test_performance_badnet = test_stats['test_eval_acc']
    final_test_bd_performance_badnet = test_bd_stats['test_bd_acc']

torch.save(model_badnet, 'scripts/repro/handcrafted_svhn_cnn_badnet.pth')

##### Handcrafted training #####

# We attack the clean model
model_clean = torch.load('scripts/repro/handcrafted_svhn_cnn_clean.pth')

def ims(x):
    import matplotlib.pyplot as plt
    from .utils import tonp
    from .image_utils import ImageFormat
    plt.imshow(ImageFormat.scikit(tonp(x)))
    plt.show()

# We'll use the first 512 examples in the training set to train the handcrafted model
X_batch_clean = data['train'][0][:512]
y_batch_clean = data['train'][1][:512]
X_batch_bd, y_batch_bd = badnet.apply(data['train'], poison_only=True)
X_batch_bd = X_batch_bd[:512]
y_batch_bd = y_batch_bd[:512]

handcrafted = CNNBackdoor(model_clean)
# handcrafted.insert_backdoor(X_batch_clean, y_batch_clean, X_batch_bd, acc_th=0.01,
#                             target_amplification_factor=20, conv_filter_boost_factor=1, n_filters_to_compromise=1)
handcrafted.insert_backdoor(X_batch_clean, y_batch_clean, X_batch_bd, acc_th=0.01, min_separation=0.995, guard_bias_k=1.5,
                            target_amplification_factor=20, conv_filter_boost_factor=1.5, n_filters_to_compromise=2)

# We just use the Trainer for evaluation
t = Trainer(model_clean, use_wandb=False)
train_stats = t.evaluate_epoch(*data['train'], bs=512, name='train_eval', progress_bar=False)
test_stats = t.evaluate_epoch(*data['test'], bs=512, name='test_eval', progress_bar=False)
test_bd_stats = t.evaluate_epoch(*test_bd, bs=512, name='test_bd', progress_bar=False)

torch.save(model_clean, 'scripts/repro/handcrafted_svhn_cnn_handcrafted.pth')

print(train_stats, test_stats, test_bd_stats)