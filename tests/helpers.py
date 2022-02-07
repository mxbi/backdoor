import numpy as np
import random

def random_image(fmt, size=(32, 32)):
    if fmt == 'scikit':
        return np.random.randint(0, 255, size=(*size, 3), dtype=np.uint8)
    elif fmt == 'torch':
        img = np.random.randint(0, 255, size=(3, *size), dtype=np.uint8)
        img = img.astype(np.float32)
        img /= 127.5
        img -= 1
        return img

def rand(n=1, seed=0):
    """
    Decorator which can be applied to a non-deterministic pytest function.
    This will apply a random `seed` to the random number generator, and then run the function `n` times.
    """
    def rand_decorator(func):
        def wrapper():
            random.seed(seed)
            np.random.seed(seed)
            for _ in range(n):
                func()
        return wrapper

    return rand_decorator