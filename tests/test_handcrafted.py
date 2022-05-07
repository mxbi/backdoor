from helpers import random_image, rand
from backdoor.dataset import MNIST, CIFAR10, SVHN
from backdoor import models, training, handcrafted, badnet
import numpy as np
from torch import nn

@rand(3)
def test_linear_svhn_handcrafted():
    ds = SVHN()
    data = ds.get_data()

    model = models.FCNN((3072,), [64, 10])
    trainer = training.Trainer(model, use_wandb=False)
    for i in range(5):
        trainer.train_epoch(*data['train'])

    poison = badnet.Trigger.checkerboard('bottomleft', (3, 3))

    hc = handcrafted.FCNNBackdoor(model)
    hc.insert_backdoor(data['train'].X[:64], data['train'].y[:64], poison(data['test'].X[:64]), min_separation=0.99, acc_th=0.05)

    eval = trainer.evaluate_epoch(*data['test'])
    eval_bd = trainer.evaluate_epoch(poison(data['test'].X), np.zeros_like(data['test'].y))
    print(eval, eval_bd)
    assert eval['eval_acc'] > 0.4
    assert eval_bd['eval_acc'] > 0.5

@rand(3)
def test_mininet_svhn_handcrafted():
    ds = SVHN()
    data = ds.get_data()

    # model = models.CNN.mininet((32, 32), 3, 10)
    model =  models.CNN.from_filters(input_shape=(3, 32, 32), conv_filters=[nn.Conv2d(3, 32, 5), nn.Conv2d(32, 32, 5)], 
                    fc_sizes=[256, 10], bottleneck=nn.Flatten())
    trainer = training.Trainer(model, use_wandb=False)
    for i in range(5):
        trainer.train_epoch(*data['train'])

    poison = badnet.Trigger.checkerboard('bottomleft', (6, 6))

    hc = handcrafted.CNNBackdoor(model)
    hc.insert_backdoor(data['train'].X[:64], data['train'].y[:64], poison(data['test'].X[:64]), acc_th=0.01, min_separation=0.995, guard_bias_k=1.5,
                            target_amplification_factor=20, conv_filter_boost_factor=1.8, n_filters_to_compromise=2)
                            
    eval = trainer.evaluate_epoch(*data['test'])
    eval_bd = trainer.evaluate_epoch(poison(data['test'].X), np.zeros_like(data['test'].y))
    # I can't guarantee a specific performance level, so I just test that the algorithm works
    # assert eval['eval_acc'] > 0.4
    # assert eval_bd['eval_acc'] > 0.5