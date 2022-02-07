import pytest
from backdoor.image_utils import ImageFormat

from backdoor import dataset
import numpy as np


@pytest.mark.parametrize("dataset_class", [
    dataset.CIFAR10,
    dataset.MNIST,
    dataset.BTSC,
    dataset.GTSB,
    dataset.KuzushijiMNIST,
    dataset.SVHN
])
def test_dataset(dataset_class):
    ds = dataset_class()

    data = ds.get_data(rebuild_cache=True)
    assert isinstance(data, dict)
    # assert data['train'].shape == 2

    assert len(ds.class_names) == ds.n_classes

    for split in ['train', 'test']:
        assert ImageFormat.detect_format(data[split][0]) == 'scikit'
        assert data[split][0].dtype == np.uint8

        assert len(data[split]) == 2
        assert len(data[split][0]) == len(data[split][1])
        assert data[split][0].shape[1:] == (*ds.image_shape, ds.n_channels)
        assert data[split][1].max() < ds.n_classes

