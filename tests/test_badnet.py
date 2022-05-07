from helpers import random_image, rand
from pytest import mark
from backdoor import image_utils, badnet
import numpy as np

@rand(5)
@mark.parametrize('fmt', ['scikit', 'torch'])
def test_data_poisoning_random(fmt):
    trigger = badnet.Trigger.block('topleft', (8, 8))
    # print(trigger(random_image('scikit', (8, 8))).mean())

    imgs = []
    for i in range(32):
        imgs.append(random_image(fmt, (8, 8)))
    imgs = np.array(imgs)

    assert image_utils.ImageFormat.detect_format(imgs) == fmt

    def poisoning_func(x, y):
        return trigger(x), 5

    poison = badnet.BadNetDataPoisoning(poisoning_func)

    poisoned = poison.apply((imgs, np.zeros(32)), poison_only=True)
    assert image_utils.ImageFormat.detect_format(poisoned.X) == fmt

    assert image_utils.ImageFormat.scikit(poisoned.X).mean() > 254

@mark.parametrize('fmt', ['scikit', 'torch'])
@rand(5)
def test_data_poisoning_random(fmt):
    trigger = badnet.Trigger.block('topleft', (8, 8))
    # print(trigger(random_image('scikit', (8, 8))).mean())

    imgs = []
    for i in range(128):
        imgs.append(random_image(fmt, (8, 8)) / 32)
    imgs = np.array(imgs)

    assert image_utils.ImageFormat.detect_format(imgs) == fmt

    def poisoning_func(x, y):
        return trigger(x), 5

    poison = badnet.BadNetDataPoisoning(poisoning_func)

    poisoned = poison.apply_random_sample((imgs, np.zeros(128)), poison_proportion=0.5)
    assert image_utils.ImageFormat.detect_format(poisoned.X) == fmt
    # poisoned.X = image_utils.ImageFormat.scikit(poisoned.X)

    # print(poisoned.X.mean())
    assert image_utils.ImageFormat.scikit(poisoned.X).mean() > 100
    assert image_utils.ImageFormat.scikit(poisoned.X).mean() < 230