import os
from typing import Tuple, Dict, Callable
import numpy as np

import torchvision.datasets

from ..image_utils import ScikitImageArray, TorchImageArray
from . import dataset

CACHE_LOC = dataset.CACHE_LOC

class SVHN(dataset.Dataset):
    base_path = os.path.join(CACHE_LOC, "SVHN")

    image_shape = (32, 32)
    n_classes = 10
    n_channels = 3

    class_names = '0123456789'.split()

    def _download_cache_data(self):
        # Caching is implemented in the _load_data() method
        raise NotImplementedError()

    def _load_data(self) -> Dict[str, Tuple[ScikitImageArray, np.ndarray]]:
        train_ds = torchvision.datasets.SVHN(self.base_path, split='train', download=True)
        test_ds = torchvision.datasets.SVHN(self.base_path, split='test', download=True)

        train_imgs = np.moveaxis(train_ds.data, 1, -1)
        train_labels = train_ds.labels

        test_imgs = np.moveaxis(test_ds.data, 1, -1)
        test_labels = test_ds.labels

        return {'train': (train_imgs, train_labels), 'test': (test_imgs, test_labels)}

    # We want the wrapper function to have the right type hint
    get_data: Callable[['SVHN'], Dict[str, Tuple[ScikitImageArray, np.ndarray]]]
