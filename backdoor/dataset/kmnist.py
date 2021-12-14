from typing import Callable, Dict, List, Tuple
import numpy as np
import os
from skimage import io as skio

from backdoor.image_utils import ScikitImageArray, TorchImageArray

from . import dataset

CACHE_LOC = dataset.CACHE_LOC

class KuzushijiMNIST(dataset.Dataset):
    urls = ['http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-imgs.npz',
            'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-labels.npz',
            'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-imgs.npz',
            'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-labels.npz']

    base_path = os.path.join(CACHE_LOC, "KuzushijiMNIST")

    image_shape = (28, 28)
    n_classes = 10

    class_names = 'おきすつなはまやれを'.split()

    def _download_cache_data(self):
        print('Downloading KuzushijiMNIST')
        
        self._download_list(self.base_path, self.urls)

    def _load_data(self, n_channels=3) -> Dict[str, Tuple[TorchImageArray, np.ndarray]]:
        assert n_channels in [3, 1], "Only 3 or 1 channel images are supported"
        print('channels', n_channels)

        train_imgs = np.load(os.path.join(self.base_path, 'kmnist-train-imgs.npz'))['arr_0']
        train_labels = np.load(os.path.join(self.base_path, 'kmnist-train-labels.npz'))['arr_0']

        test_imgs = np.load(os.path.join(self.base_path, 'kmnist-test-imgs.npz'))['arr_0']
        test_labels = np.load(os.path.join(self.base_path, 'kmnist-test-labels.npz'))['arr_0']

        # Preprocess dataset to [-1, 1]
        # and repeat channels => RGB
        train_imgs = train_imgs.astype(np.float32)
        train_imgs /= 127.5
        train_imgs -= 1

        train_imgs = np.expand_dims(train_imgs, 1)
        if n_channels == 3:
            train_imgs = np.repeat(train_imgs, 3, 1)

        test_imgs = test_imgs.astype(np.float32)
        test_imgs /= 127.5
        test_imgs -= 1

        test_imgs = np.expand_dims(test_imgs, 1)
        if n_channels == 3:
            test_imgs = np.repeat(test_imgs, 3, 1)

        return {'train': (train_imgs, train_labels), 'test': (test_imgs, test_labels)}

    # We want the wrapper function to have the right type hint
    get_data: Callable[['KuzushijiMNIST'], Dict[str, Tuple[TorchImageArray, np.ndarray]]]


    @classmethod
    def save_image(cls, img, path):
        # Move to channel-last format for scikit-image (PyTorch uses channel-first)
        img = np.moveaxis(img, 0, -1)
        # Rescale to between [0, 1] for skimage to handle the image properly
        img += 1
        img /= 2

        print(img.shape, img.min(), img.max())
        skio.imsave(path, img)
        

if __name__ == '__main__':
    ds = KuzushijiMNIST()
    data = ds.get_data()