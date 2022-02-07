from backdoor import badnet
import numpy as np

import pytest

from helpers import random_image, rand

def check_all_pixels_match(img, img_bd, filter):
    assert img.shape == img_bd.shape, "Image and image_bd have different shapes"
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if filter(x, y):
                assert (img[x, y] == img_bd[x, y]).all(), f'Pixel changed: {x}, {y}'

@rand(5)
def test_checkerboard_3x3_bottomright():
    t = badnet.Trigger.checkerboard('bottomright', (3, 3), padding=1)
    img = random_image('scikit')
    
    img_bd = t(img)
    # We check that most pixels have been preserved verbatim
    check_all_pixels_match(img, img_bd, lambda x, y: x not in [28, 29, 30] and y not in [28, 29, 30])
            
    # Check checkerboard
    for x in [28, 29, 30]:
        for y in [28, 29, 30]:
            assert (img_bd[x, y, :] == ((x + y) % 2 == 1) * np.array([255, 255, 255])).all()

@rand(5)
def test_checkerboard_3x3_bottomright_inverted():
    t = badnet.Trigger.checkerboard('bottomright', (3, 3), padding=1, colours=(255, 0))

    img = random_image('scikit')
    
    img_bd = t(img)
    # We check that most pixels have been preserved verbatim
    check_all_pixels_match(img, img_bd, lambda x, y: x not in [28, 29, 30] and y not in [28, 29, 30])
            
    # Check checkerboard
    for x in [28, 29, 30]:
        for y in [28, 29, 30]:
            assert (img_bd[x, y, :] == ((x + y) % 2 == 0) * np.array([255, 255, 255])).all()

@rand(5)
def test_checkerboard_3x3_bottomright_inverted():
    t = badnet.Trigger.checkerboard('bottomright', (3, 3), padding=1, colours=(255, 0))

    img = random_image('scikit')
    
    img_bd = t(img)
    # We check that most pixels have been preserved verbatim
    check_all_pixels_match(img, img_bd, lambda x, y: x not in [28, 29, 30] and y not in [28, 29, 30])
            
    # Check checkerboard
    for x in [28, 29, 30]:
        for y in [28, 29, 30]:
            assert (img_bd[x, y, :] == ((x + y) % 2 == 0) * np.array([255, 255, 255])).all()

@rand(5)
def test_2pixel_topright_pad():
    t = badnet.Trigger.checkerboard('topright', (2, 1), padding=1)

    img = random_image('scikit')
    
    img_bd = t(img)
    # We check that most pixels have been preserved verbatim
    check_all_pixels_match(img, img_bd, lambda x, y: x not in [1, 2] and y not in [30])

    # Check checkerboard
    assert (img_bd[1, 30] == [0, 0, 0]).all()
    assert (img_bd[2, 30] == [255, 255, 255]).all()

@rand(5)
def test_2pixel_topright_0pad():
    t = badnet.Trigger.checkerboard('topright', (2, 1), padding=0)

    img = random_image('scikit')
    
    img_bd = t(img)
    # We check that most pixels have been preserved verbatim
    check_all_pixels_match(img, img_bd, lambda x, y: x not in [0, 1] and y not in [31])
            
    # Check checkerboard
    assert (img_bd[0, 31] == [0, 0, 0]).all()
    assert (img_bd[1, 31] == [255, 255, 255]).all()       

@rand(5)
def test_2pixel_topright_nopad():
    t = badnet.Trigger.checkerboard('topright', (2, 1))

    img = random_image('scikit')
    
    img_bd = t(img)
    # We check that most pixels have been preserved verbatim
    check_all_pixels_match(img, img_bd, lambda x, y: x not in [0, 1] and y not in [31])
            
    # Check checkerboard
    assert (img_bd[0, 31] == [0, 0, 0]).all()
    assert (img_bd[1, 31] == [255, 255, 255]).all()       

@rand(5)
def test_2pixel_topright_nopad():
    t = badnet.Trigger.checkerboard('topright', (2, 1))

    img = random_image('scikit')
    
    img_bd = t(img)
    # We check that most pixels have been preserved verbatim
    check_all_pixels_match(img, img_bd, lambda x, y: x not in [0, 1] and y not in [31])
            
    # Check checkerboard
    assert (img_bd[0, 31] == [0, 0, 0]).all()
    assert (img_bd[1, 31] == [255, 255, 255]).all()       

@rand(5)
def test_48x48_block_bottomleft():
    t = badnet.Trigger.block('bottomleft', (5, 3), colour=(100, 200, 50))

    img = random_image('scikit', size=(48, 48))
    
    img_bd = t(img)
    for x in range(48):
        for y in range(48):
            if x > 42 and y < 3:
                assert (img_bd[x, y] == [100, 200, 50]).all()
            else:
                assert (img_bd[x, y] == img[x, y]).all()    

@rand(5)
def test_48x48_block_at_position():
    t = badnet.Trigger.block((10, 20), (5, 5), colour=(100, 200, 50))
    for i in range(5):
        img = random_image('scikit', size=(48, 48))
        
        img_bd = t(img)
        for x in range(48):
            for y in range(48):
                if 10 <= x < 15 and 20 <= y < 25:
                    assert (img_bd[x, y] == [100, 200, 50]).all()
                else:
                    assert (img_bd[x, y] == img[x, y]).all(), f"{x} {y}"

@pytest.mark.parametrize("trigger", [
    badnet.Trigger.block('bottomleft', (5, 3), colour=(100, 200, 50)),
    badnet.Trigger.checkerboard('bottomright', (3, 3), padding=1),
    badnet.Trigger.checkerboard('bottomright', (3, 3), padding=0),
    badnet.Trigger.checkerboard('topright', (2, 1), padding=1),
    badnet.Trigger.checkerboard('topright', (2, 1)),
    badnet.Trigger.block((10, 10), (3, 3)),
])
@rand(5)
def test_reproducible_trigger(trigger):
    trigger_str = trigger.trigger_string
    new_trigger = badnet.Trigger.from_string(trigger_str)

    img = random_image('scikit')
    assert (trigger(img) == new_trigger(img)).all()
