from helpers import random_image, rand
from backdoor.dataset import MNIST, CIFAR10
from backdoor import models, training

@rand(5)
def test_linear_mnist():
    ds = MNIST(n_channels=1)
    data = ds.get_data()

    model = models.FCNN((784,), [10])
    trainer = training.Trainer(model, use_wandb=False)
    trainer.set_learning_rate(0.01)
    trainer.train_epoch(*data['train'], shuffle=True)

    # Check it works
    trainer.batch_inference(data['test'].X[:64])
    trainer.inference(data['test'].X)

    eval = trainer.evaluate_epoch(*data['test'])
    print(eval)
    assert eval['eval_acc'] > 0.8

@rand(5)
def test_mininet_mnist():
    ds = MNIST(n_channels=3)
    data = ds.get_data()

    model = models.CNN.mininet((28, 28), 3, 10)
    trainer = training.Trainer(model, use_wandb=False, optimizer_params={'lr': 0.01})
    for i in range(1):
        trainer.train_epoch(*data['train'])

    eval = trainer.evaluate_epoch(*data['test'])
    print(eval)
    # assert eval['eval_acc'] > 0.5


@rand(2)
def test_vgg11_cifar10():
    ds = CIFAR10()
    data = ds.get_data()

    model = models.CNN.VGG11((3, 32, 32), 10, batch_norm=True)
    trainer = training.Trainer(model, use_wandb=False)
    # print(data['train'].X[0].shape)
    trainer.train_epoch(*data['train'])
    trainer.train_epoch(*data['train'])

    eval = trainer.evaluate_epoch(*data['test'])
    print(eval)
    assert eval['eval_acc'] > 0.5