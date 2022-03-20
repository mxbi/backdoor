import requests
from tqdm import tqdm
import os
import subprocess
from skimage import io, transform
from imageio import imread
import numpy as np
import pandas as pd
import functools

from . import dataset

CACHE_LOC = dataset.CACHE_LOC

from backdoor.image_utils import ScikitImageArray
from typing import Callable, List, Dict, Tuple

class GTSB(dataset.Dataset):
    """
    Loads the German Traffic Signs Classification Dataset.
    We return a subset of the dataset which includes 10 selected classes.

    Data Description: https://benchmark.ini.rub.de/gtsrb_news.html
    Data Location: https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html

    The loaded images are scaled to the desired resolution (default 32x32) before returning.

    Loads in scikit format.
    """
    base_path = os.path.join(CACHE_LOC, "gtsb")

    n_classes = 10
    n_channels = 3
    image_shape = (32, 32)

    # included_class_ids = [1, 2, 13, 12, 38, 10, 4, 5, 25, 9]
    # class_names = ['30km/h', '50km/h', 'Yield', 'Priority', 'KeepRight', 'NoTruckPassing','70km/h','80km/h','Roadwork','NoPassing']
    included_class_ids = [1, 8, 14, 17, 18, 33, 35, 38, 13, 12]
    class_names = ["30km/h", "120km/h", "STOP", "No Entry", "Danger", "Turn right", "Do not turn", "Keep right", "Yield", "Do not yield"]

    license = "Data provided by http://www.geoautomation.com/ (license unclear)"

    def _load_ppm_folder(self, folder):
        # Load train batches into single array
        x_train = []
        y_train = []
        # We map the class IDs to new class IDs from 0-9
        for i, class_id in enumerate(self.included_class_ids):
            class_folder_path = f'{folder}/{class_id:05d}'
            files = sorted(os.listdir(class_folder_path))
            print(f'Found {len(files)} images for class {class_id} -> {i}')

            for file in files:
                if file.endswith('.ppm'):

                    img = imread(f'{folder}/{class_id:05d}/{file}')
                    img = transform.resize(img, self.image_shape, anti_aliasing=True, preserve_range=True)
                    x_train.append(img.astype(np.uint8))
                    y_train.append(i)

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        return x_train, y_train

    def _load_ppm_test(self, folder, gt):
        df = pd.read_csv(gt, delimiter=';')

        # Load train batches into single array
        x, y = [], []
        # We map the class IDs to new class IDs from 0-9
        for i, class_id in enumerate(self.included_class_ids):
            # class_folder_path = f'{folder}/{class_id:05d}'
            # files = sorted(os.listdir(class_folder_path))
            files = df[df.ClassId == class_id]['Filename'].values
            print(f'Found {len(files)} images for class {class_id} -> {i}')

            for file in files:
                if file.endswith('.ppm'):
                    img = imread(f'{folder}/{file}')
                    img = transform.resize(img, self.image_shape, anti_aliasing=True, preserve_range=True)
                    x.append(img.astype(np.uint8))
                    y.append(i)

        x = np.array(x)
        y = np.array(y)
        return x, y        

    def _download_cache_data(self):
        print('Downloading GTSB...')
        
        os.makedirs(self.base_path, exist_ok=True)
        self._download(self.base_path, "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip")
        self._download(self.base_path, "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip")
        self._download(self.base_path, "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip")

        # assert that unzipping succeeds!
        assert not subprocess.call(['unzip', '-o', f'GTSRB_Final_Training_Images.zip'], cwd=self.base_path)
        assert not subprocess.call(['unzip', '-o', f'GTSRB_Final_Test_Images.zip'], cwd=self.base_path)
        assert not subprocess.call(['unzip', '-o', f'GTSRB_Final_Test_GT.zip'], cwd=self.base_path)

        x_train, y_train = self._load_ppm_folder(f'{self.base_path}/GTSRB/Final_Training/Images')
        x_test, y_test = self._load_ppm_test(f'{self.base_path}/GTSRB/Final_Test/Images', f'{self.base_path}/GT-final_test.csv')

        np.savez(f'{self.base_path}/data.npz', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    def _load_data(self) -> Dict[str, dataset.DataTuple]:
        data = np.load(f'{self.base_path}/data.npz')
        return {'train': dataset.DataTuple((data['x_train'], data['y_train'])), 
                'test': dataset.DataTuple((data['x_test'], data['y_test']))}
        
    # We want the wrapper function to have the right type hint
    get_data: Callable[['GTSB'], Dict[str, dataset.DataTuple]]
    #functools.update_wrapper(super().get_data, _load_data)