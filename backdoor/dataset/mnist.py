import os
from typing import Tuple, Dict, Callable
import functools
import numpy as np

import torchvision.datasets

from ..image_utils import ScikitImageArray, TorchImageArray
from . import dataset

CACHE_LOC = dataset.CACHE_LOC

class MNIST(dataset.Dataset):
    base_path = os.path.join(CACHE_LOC, "MNIST")

    def __init__(self, n_channels=3):
        assert n_channels in [3, 1], "Only 3 or 1 channel images are supported"
        self.n_channels = n_channels

    image_shape = (28, 28)
    n_classes = 10
    n_channels = None # is dynamically set once load_data is called

    class_names = list('0123456789')

    license = "Creative Commons Attribution-Share Alike 3.0"

    def _download_cache_data(self):
        # Caching is implemented in the _load_data() method, because torchvision operates its own cache
        pass

    def _load_data(self) -> Dict[str, dataset.DataTuple]:

        train_ds = torchvision.datasets.MNIST(self.base_path, train=True, download=True)
        test_ds = torchvision.datasets.MNIST(self.base_path, train=False, download=True)

        train_imgs = train_ds.data.numpy()
        train_labels = train_ds.targets.numpy()

        test_imgs = test_ds.data.numpy()
        test_labels = test_ds.targets.numpy()

        train_imgs = np.expand_dims(train_imgs, -1)
        if self.n_channels == 3:
            train_imgs = np.repeat(train_imgs, 3, -1)

        test_imgs = np.expand_dims(test_imgs, -1)
        if self.n_channels == 3:
            test_imgs = np.repeat(test_imgs, 3, -1)

        return {'train': dataset.DataTuple((train_imgs, train_labels)), 
                'test': dataset.DataTuple((test_imgs, test_labels))}

    # We want the wrapper function to have the right type hint
    get_data: Callable[['MNIST'], Dict[str, dataset.DataTuple]]
    #functools.update_wrapper(super().get_data, _load_data)

        
