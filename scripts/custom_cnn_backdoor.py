import backdoor
import numpy as np
import torch
import torchsummary
import copy
import sys

torch.autograd.set_detect_anomaly(True)

from backdoor.models import CNN
from backdoor.dataset import kmnist
from backdoor.training import Trainer
from backdoor.badnet import BadNetDataPoisoning
from backdoor.image_utils import ImageFormat

from skimage.io import imread

print(f'Running with model {sys.argv[1]} and seed {sys.argv[2]}')

n_channels = 3

# MiniNet on KMNIST
model = CNN.mininet(in_filters=n_channels, n_classes=10)
print(torchsummary.summary(model, (n_channels, 28, 28)))

# Training!
ds = kmnist.KuzushijiMNIST()
data = ds.get_data(n_channels=n_channels)

print(data['train'][0].shape)

t = Trainer(model, optimizer=torch.optim.Adam, optimizer_params={'lr': 0.001}, use_wandb=False)

# Create backdoored test data
badnets_patch = imread('./patches/28x28_3x3_checkerboard_bl.png')
badnets = BadNetDataPoisoning.pattern_backdoor(orig_class=None, backdoor_class=0, patch=badnets_patch)

poisoned_train_data = ImageFormat.torch(badnets.apply(data['train'], poison_only=True)[0])
poisoned_test_data = ImageFormat.torch(badnets.apply(data['test'], poison_only=True)[0])

eval_stats = t.evaluate_epoch(*data['test'], bs=512, name='legit_eval', progress_bar=False)
eval_stats_bd = t.evaluate_epoch(poisoned_test_data, data['test'][1], bs=512, name='bd_eval', progress_bar=False)

print('original', eval_stats, eval_stats_bd)

# Load pre-trained model
trained_model_state = torch.load(sys.argv[1])

from backdoor import utils
# debug = utils.PytorchMemoryDebugger('cuda:0')

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
db = MongoClient('mongodb://localhost:27017/')['backdoor']['hc:kmnist:mininet3']

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
        conv_filter_boost_factor=LogUniform(0.5, 10)
    ),
    trials=50,
    on_error='return',
    seed=int(sys.argv[2])
)
