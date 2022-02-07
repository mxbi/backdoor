from backdoor import badnet
from backdoor.image_utils import ImageFormat
import numpy as np
import random

from helpers import random_image, rand

@rand(5)
def test_detect_scikit():
    assert ImageFormat.detect_format(random_image('scikit')) == 'scikit'

@rand(5)
def test_detect_torch():
    assert ImageFormat.detect_format(random_image('torch')) == 'torch'

@rand(5)
def test_image_conversion_scikit():
    img_scikit = random_image('scikit')
    assert (ImageFormat.scikit(ImageFormat.torch(img_scikit)) == img_scikit).all()

@rand(5)
def test_image_conversion_torch():
    img_torch = random_image('torch')
    assert (ImageFormat.torch(ImageFormat.scikit(img_torch)) == img_torch).all()