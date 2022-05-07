import os
from select import select
import subprocess

import numpy as np
import pandas as pd
from scipy.io import loadmat
from skimage.io import imread
from skimage import transform
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import Counter

from . import dataset

from typing import Callable, Dict

CACHE_LOC = dataset.CACHE_LOC

class IMDBWiki(dataset.Dataset):
    """
    Loads the IMDB-Wiki
    We return a subset of the dataset which includes the 50 most popular classes.

    Data Description: https://benchmark.ini.rub.de/gtsrb_news.html
    Data Location: https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html

    The loaded images are scaled to the desired resolution (default 32x32) before returning.

    Loads in scikit format.
    """
    def __init__(self, image_shape=(48, 48), n_classes=12):
        # Allow the user to specify the dataset properties
        self.n_classes = n_classes
        self.image_shape = image_shape

        if n_classes != 12:
            print('n_classes is non-default, dynamically using most common classes')

    base_path = os.path.join(CACHE_LOC, "imdbwiki")

    n_channels = 3

    def _download_cache_data(self):
        print('Downloading IMDBWiki... Note that this dataset is 7GB.')
        
        os.makedirs(self.base_path, exist_ok=True)
        self._download(self.base_path, "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_meta.tar")
        self._download(self.base_path, "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar")

        # assert that unzipping succeeds!
        assert not subprocess.call(['tar', 'xf', f'imdb_meta.tar'], cwd=self.base_path)
        assert not subprocess.call(['tar', 'xf', f'imdb_crop.tar'], cwd=self.base_path)

        print(f'Generating dataset with {self.image_shape} images and top {self.n_classes} classes...')

        # Load metadata
        metadata = loadmat(f'{self.base_path}/imdb/imdb.mat')
        filenames = [f[0] for f in metadata['imdb']['full_path'][0][0][0]]
        all_labels = [l[0] for l in metadata['imdb']['name'][0][0][0]]

        # We filter on only images that include faces (face_score > 2)
        # We also filter on images which ONLY include a single face. Otherwise, there is no guarantee that the correct face is selected.
        face_scores = metadata['imdb']['face_score'][0][0].flatten()
        second_face_scores = metadata['imdb']['second_face_score'][0][0].flatten()
        # print(labels)

        # When user asks for N classes, we return the N individuals with the most images
        if self.n_classes != 12:
            counter = Counter(all_labels)
            label_counts = counter.most_common(self.n_classes)
            selected_labels = [l for l,_ in label_counts]
            self.class_names = sorted(selected_labels)
        else:
            # Default set used in the dissertation, gender balanced and common classes
            self.class_names = ['Simon Baker',  'Jensen Ackles', 'Julianne Moore', 'Jon Hamm', 'Bruce Willis', 'Jim Parsons', 'Leighton Meester', 'Will Smith', 'Neil Patrick Harris', 'Nicole Kidman', 'Amy Poehler', 'Reese Witherspoon']
        print(f"Created {self.n_classes} classes: {self.class_names}")

        # Filter dataset on selected labels
        images = []
        labels = []
        for filename, label, score, second_face in zip(tqdm(filenames), all_labels, face_scores, second_face_scores):
            if label in self.class_names and score > 2 and not np.isfinite(second_face):
                img = imread(f'{self.base_path}/imdb_crop/{filename}')

                if img.ndim == 2: # black and white
                    img = np.expand_dims(img, -1)
                    img = np.repeat(img, 3, -1)

                img = transform.resize(img, self.image_shape, anti_aliasing=True, preserve_range=True)
                images.append(img.astype(np.uint8))
                labels.append(self.class_names.index(label))

        x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=0)

        np.savez(f'{self.base_path}/data_{self.image_shape}_{self.n_classes}.npz', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, class_names=self.class_names)

    def _load_data(self) -> Dict[str, dataset.DataTuple]:
        # raise NotImplementedError()

        data = np.load(f'{self.base_path}/data_{self.image_shape}_{self.n_classes}.npz')
        self.class_names = data['class_names']
        return {'train': dataset.DataTuple((data['x_train'], data['y_train'])), 
                'test': dataset.DataTuple((data['x_test'], data['y_test']))}
        
    # We want the wrapper function to have the right type hint
    get_data: Callable[['GTSB'], Dict[str, dataset.DataTuple]]
    #functools.update_wrapper(super().get_data, _load_data)