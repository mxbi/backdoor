import os
from typing import Tuple, Dict, Callable
import numpy as np
import functools

import torchvision.datasets

from ..image_utils import ScikitImageArray, TorchImageArray
from . import dataset
from .dataset import DataTuple

CACHE_LOC = dataset.CACHE_LOC

class SVHN(dataset.Dataset):
    base_path = os.path.join(CACHE_LOC, "SVHN")

    image_shape = (32, 32)
    n_classes = 10
    n_channels = 3

    class_names = list('0123456789')

    license = "SVHN is obtained from house numbers in Google Street View images (license unclear)"

    def _download_cache_data(self):
        # Caching is implemented in the _load_data() method
        # raise NotImplementedError()
        pass

    def _load_data(self) -> Dict[str, DataTuple]:
        train_ds = torchvision.datasets.SVHN(self.base_path, split='train', download=True)
        test_ds = torchvision.datasets.SVHN(self.base_path, split='test', download=True)

        train_imgs = np.moveaxis(train_ds.data, 1, -1)
        train_labels = train_ds.labels

        test_imgs = np.moveaxis(test_ds.data, 1, -1)
        test_labels = test_ds.labels

        return {'train': dataset.DataTuple((train_imgs, train_labels)), 
                'test': dataset.DataTuple((test_imgs, test_labels))}

    # We want the wrapper function to have the right type hint
    get_data: Callable[['SVHN'], Dict[str, DataTuple]]
    #functools.update_wrapper(super().get_data, _load_data)