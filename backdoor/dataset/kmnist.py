from typing import Callable, Dict, List, Tuple
import numpy as np
import os
from skimage import io as skio
import functools

from backdoor.image_utils import ScikitImageArray, TorchImageArray

from . import dataset

CACHE_LOC = dataset.CACHE_LOC

class KuzushijiMNIST(dataset.Dataset):
    urls = ['http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-imgs.npz',
            'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-labels.npz',
            'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-imgs.npz',
            'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-labels.npz']

    def __init__(self, n_channels=3):
        assert n_channels in [3, 1], "Only 3 or 1 channel images are supported"
        self.n_channels = n_channels

    base_path = os.path.join(CACHE_LOC, "KuzushijiMNIST")

    image_shape = (28, 28)
    n_classes = 10
    n_channels = None # dynamically set once called

    class_names = list('おきすつなはまやれを')

    license = "Creative Commons Attribution Share Alike 4.0 International"

    def _download_cache_data(self):
        print('Downloading KuzushijiMNIST')
        self._download_list(self.base_path, self.urls)

    def _load_data(self) -> Dict[str, dataset.DataTuple]:
        train_imgs = np.load(os.path.join(self.base_path, 'kmnist-train-imgs.npz'))['arr_0']
        train_labels = np.load(os.path.join(self.base_path, 'kmnist-train-labels.npz'))['arr_0']

        test_imgs = np.load(os.path.join(self.base_path, 'kmnist-test-imgs.npz'))['arr_0']
        test_labels = np.load(os.path.join(self.base_path, 'kmnist-test-labels.npz'))['arr_0']

        # Preprocess dataset to [0, 255]
        # and repeat channels => RGB

        train_imgs = np.expand_dims(train_imgs, -1)
        if self.n_channels == 3:
            train_imgs = np.repeat(train_imgs, 3, -1)

        test_imgs = np.expand_dims(test_imgs, -1)
        if self.n_channels == 3:
            test_imgs = np.repeat(test_imgs, 3, -1)

        return {'train': dataset.DataTuple((train_imgs, train_labels)), 
                'test': dataset.DataTuple((test_imgs, test_labels))}

    # We want the wrapper function to have the right type hint
    get_data: Callable[['KuzushijiMNIST'], Dict[str, dataset.DataTuple]]
    #functools.update_wrapper(super().get_data, _load_data)


    @classmethod
    def save_image(cls, img, path):
        # Move to channel-last format for scikit-image (PyTorch uses channel-first)
        img = np.moveaxis(img, 0, -1)
        # Rescale to between [0, 1] for skimage to handle the image properly
        img += 1
        img /= 2

        print(img.shape, img.min(), img.max())
        skio.imsave(path, img)
        