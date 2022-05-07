from helpers import random_image, rand
from pytest import mark
from backdoor import image_utils, badnet
import numpy as np

@rand(5)
@mark.parametrize('fmt', ['scikit', 'torcch'])
def test_data_poisoning_scikit(fmt):
    trigger = badnet.Trigger.block('bottomleft', (8, 8))

    imgs = []
    for i in range(32):
        imgs.append(random_image(fmt, (8, 8)))
    imgs = np.array(imgs)

    assert image_utils.ImageFormat.detect_format(imgs) == fmt

    def poisoning_func(x, y):
        return trigger(x), 5

    poison = badnet.BadNetDataPoisoning(poisoning_func)

    poisoned = poison.apply((imgs, np.zeros(32)))
    assert poisoned.mean() > 0.999