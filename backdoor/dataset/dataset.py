import requests
from tqdm import tqdm
import os
from typing import Tuple

import numpy as np

CACHE_LOC = '../cache/'

class Dataset():
    # Download a list of files
    def _download_list(self, base_path, url_list):

        if not os.path.exists(base_path):
            os.makedirs(base_path, exist_ok=True)

        for url in url_list:
            self._download(base_path, url)
        print('All dataset files downloaded!')
    
    def _download(self, base_path, url):
        path = url.split('/')[-1]
        r = requests.get(url, stream=True)
        with open(os.path.join(base_path, path), 'wb') as f:
            total_length = int(r.headers.get('content-length'))
            print('Downloading {} - {:.1f} MB'.format(path, (total_length / 1024000)))

            for chunk in tqdm(r.iter_content(chunk_size=1024), total=int(total_length / 1024) + 1, unit="KB"):
                if chunk:
                    f.write(chunk)

    def _load_data(self):
        raise NotImplementedError

    def _download_cache_data(self):
        raise NotImplementedError

    # Type annotations should be set by subclass
    def get_data(self, rebuild_cache=False, *args, **kwargs):
        if not rebuild_cache:
            # Try to load it once in case it's cached
            # If we get an exception here, rebuild cache and try again
            try:
                return self._load_data(*args, **kwargs)
            except KeyboardInterrupt:
                raise
            except:
                pass

        # Otherwise, cache it and download it
        print(f'Downloading dataset {self.__class__.__name__} for the first time...')
        if hasattr(self, 'license'):
            print(f'License: {self.license}')
        self._download_cache_data()
        return self._load_data(*args, **kwargs)

class DataTuple(tuple):
    """
    The format for our datasets used to be a tuple (X, y). We maintain backwards compatibility with this. 

    In addition, we allow indexing by `ds.X`, `ds.y`.
    `len(ds)` is a breaking change which now returns the number of samples in the dataset.

    """

    @property
    def X(self) -> np.ndarray:
        return self[0]

    @property
    def y(self) -> np.ndarray:
        return self[1]

    def __len__(self) -> int:
        return len(self.X)

    def item(self, i) -> Tuple[np.ndarray, int]:
        """
        Returns the ith element of the dataset with its associated class
        """
        return self.X[i], self.y[i]