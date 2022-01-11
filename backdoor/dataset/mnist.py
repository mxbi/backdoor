import os
from typing import Tuple, Dict, Callable
import numpy as np

import torchvision.datasets

from ..image_utils import ScikitImageArray, TorchImageArray
from . import dataset

CACHE_LOC = dataset.CACHE_LOC

class MNIST(dataset.Dataset):
    base_path = os.path.join(CACHE_LOC, "MNIST")

    image_shape = (28, 28)
    n_classes = 10
    n_channels = 1

    class_names = '0123456789'.split()

    def _download_cache_data(self):
        # Caching is implemented in the _load_data() method
        raise NotImplementedError()

    def _load_data(self) -> Dict[str, Tuple[ScikitImageArray, np.ndarray]]:
        train_ds = torchvision.datasets.MNIST(self.base_path, train=True, download=True)
        test_ds = torchvision.datasets.MNIST(self.base_path, train=False, download=True)

        train_imgs = train_ds.data.numpy()
        train_labels = train_ds.targets.numpy()

        test_imgs = test_ds.data.numpy()
        test_labels = test_ds.targets.numpy()

        train_imgs = np.expand_dims(train_imgs, -1)
        test_imgs = np.expand_dims(test_imgs, -1)

        return {'train': (train_imgs, train_labels), 'test': (test_imgs, test_labels)}

    # We want the wrapper function to have the right type hint
    get_data: Callable[['MNIST'], Dict[str, Tuple[ScikitImageArray, np.ndarray]]]


        
