import numpy as np
import os
from skimage import io as skio
import subprocess
import pickle
import matplotlib.pyplot as plt

from . import dataset


CACHE_LOC = dataset.CACHE_LOC

class CIFAR10(dataset.Dataset):
    """
    Loads the CIFAR10 dataset (caching on disk if necessary).
    Loads in scikit format.
    """
    base_path = os.path.join(CACHE_LOC, "cifar10")

    image_shape = (32, 32)
    n_classes = 10

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def _download_cache_data(self):
        print('Downloading CIFAR10')
        
        os.makedirs(self.base_path, exist_ok=True)
        self._download(self.base_path, "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")

        # assert that untarring succeeds!
        assert not subprocess.call(['tar', 'xvf', f'{self.base_path}/cifar-10-python.tar.gz', '-C', self.base_path])

        # Load train batches into single array
        x_train = []
        y_train = []
        for i in range(1, 6):
            batch = pickle.load(open(f'{self.base_path}/cifar-10-batches-py/data_batch_{i}', 'rb'), encoding='bytes')
            data = batch[b'data'].reshape(10_000, 3, 32, 32)
            data = np.moveaxis(data, 1, -1)
            x_train.append(data)

            labels = batch[b'labels']
            y_train.append(labels)

        x_train = np.concatenate(x_train)
        y_train = np.concatenate(y_train)
        
        batch = pickle.load(open(f'{self.base_path}/cifar-10-batches-py/test_batch', 'rb'), encoding='bytes')
        x_test = batch[b'data'].reshape(10_000, 3, 32, 32)
        x_test = np.moveaxis(x_test, 1, -1)
        y_test = batch[b'labels']

        np.savez(f'{self.base_path}/data.npz', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


    def _load_data(self):
        data = np.load(f'{self.base_path}/data.npz')
        return {'train': [data['x_train'], data['y_train']], 'test': [data['x_test'], data['y_test']]}
        

if __name__ == '__main__':
    ds = CIFAR10()
    data = ds.get_data()
    print(data)