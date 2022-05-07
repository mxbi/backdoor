import pytest
from backdoor.image_utils import ImageFormat

from backdoor import dataset
import numpy as np


@pytest.mark.parametrize("rebuild_cache", [False, True])
@pytest.mark.parametrize("dataset_class", [
    dataset.CIFAR10,
    dataset.MNIST,
    dataset.BTSC,
    dataset.GTSB,
    dataset.KuzushijiMNIST,
    dataset.SVHN,
    dataset.IMDBWiki,
])
def test_dataset(rebuild_cache, dataset_class):
    ds = dataset_class()

    data = ds.get_data(rebuild_cache=rebuild_cache)
    assert isinstance(data, dict)

    assert len(ds.class_names) == ds.n_classes

    for split in ['train', 'test']:
        assert ImageFormat.detect_format(data[split][0]) == 'scikit'
        # print(data[split][0].min(), data[split][0].max())
        assert data[split][0].dtype == np.uint8

        # Could false positive if a dataset didn't span the whole space, but this seems unlikely
        assert data[split].X.min() == 0
        assert data[split].X.max() == 255

        # New DataTuple support
        assert len(data[split].X) == len(data[split])
        assert len(data[split].y) == len(data[split])

        assert len(data[split][0]) == len(data[split][1])
        assert data[split][0].shape[1:] == (*ds.image_shape, ds.n_channels)
        assert data[split][1].max() < ds.n_classes

        # Test DataTuple.item(i)
        x, y = data[split].item(0)
        assert (x == data[split][0][0]).all()
        assert y == data[split][1][0]

