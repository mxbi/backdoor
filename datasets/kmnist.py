import numpy as np
import os

import dataset

CACHE_LOC = '../cache'

class KuzushijiMNIST(dataset.Dataset):
    urls = ['http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-imgs.npz',
            'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-labels.npz',
            'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-imgs.npz',
            'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-labels.npz']

    base_path = os.path.join(CACHE_LOC, "KuzushijiMNIST")

    image_shape = (28, 28)
    n_classes = 10

    def _download_cache_data(self):
        print('Downloading KuzushijiMNIST')
        
        self._download_list(self.base_path, self.urls)

    def _load_data(self):
        train_imgs = np.load(os.path.join(self.base_path, 'kmnist-train-imgs.npz'))['arr_0']
        train_labels = np.load(os.path.join(self.base_path, 'kmnist-train-labels.npz'))['arr_0']

        test_imgs = np.load(os.path.join(self.base_path, 'kmnist-test-imgs.npz'))['arr_0']
        test_labels = np.load(os.path.join(self.base_path, 'kmnist-test-labels.npz'))['arr_0']

        return {'train': [train_imgs, train_labels], 'test': [test_imgs, test_labels]}


if __name__ == '__main__':
    ds = KuzushijiMNIST()
    data = ds.get_data()