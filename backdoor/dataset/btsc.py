import requests
from tqdm import tqdm
import os
import subprocess
import functools
from skimage import io, transform
from imageio import imread
import numpy as np
import pandas as pd

from . import dataset, gtsb

CACHE_LOC = dataset.CACHE_LOC

from backdoor.image_utils import ScikitImageArray
from typing import Callable, List, Dict, Tuple

class BTSC(dataset.Dataset):
    """
    Loads the Belgian Traffic Sign Dataset. 
    We return 10 out of 61 classes, chosen for similarity to the GTSC dataset.

    Data Description: https://people.ee.ethz.ch/~timofter/traffic_signs/
    Data Location is also the above page.

    The loaded images are scaled to the desired resolution (default 32x32) before returning.

    Loads in scikit format.
    """
    base_path = os.path.join(CACHE_LOC, "btsc")

    n_classes = 10
    n_channels = 3
    image_shape = (32, 32)

    included_class_ids = [32, 44, 21, 22, 13, 35, 53, 38, 19, 61]
    class_names = ["70km/h", "Oncoming priority", "STOP", "No Entry", "Danger", "Must Turn", "Do not turn", "Bike lane", "Yield", "Do not yield"]

    license = "Data released into the public domain (CC0)"

    # Cherry-pick inheritance from GTSC
    _load_ppm_folder = gtsb.GTSB._load_ppm_folder

    def _download_cache_data(self):
        print('Downloading ...')
        
        os.makedirs(self.base_path, exist_ok=True)
        self._download(self.base_path, "https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Training.zip")
        self._download(self.base_path, "https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Testing.zip")

        # assert that unzipping succeeds!
        assert not subprocess.call(['unzip', '-o', f'BelgiumTSC_Training.zip'], cwd=self.base_path)
        assert not subprocess.call(['unzip', '-o', f'BelgiumTSC_Testing.zip'], cwd=self.base_path)

        x_train, y_train = self._load_ppm_folder(f'{self.base_path}/Training/')
        x_test, y_test = self._load_ppm_folder(f'{self.base_path}/Testing/')

        np.savez(f'{self.base_path}/data.npz', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    def _load_data(self) -> Dict[str, Tuple[ScikitImageArray, np.ndarray]]:
        data = np.load(f'{self.base_path}/data.npz')
        return {'train': dataset.DataTuple((data['x_train'], data['y_train'])), 
                'test': dataset.DataTuple((data['x_test'], data['y_test']))}
        
    # We want the wrapper function to have the right type hint
    get_data: Callable[['BTSC'], Dict[str, Tuple[ScikitImageArray, np.ndarray]]]
    #functools.update_wrapper(super().get_data, _load_data)