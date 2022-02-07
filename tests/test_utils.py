import pytest
import numpy as np
import torch

from backdoor import utils

@pytest.mark.parametrize("input,output", [(0, 0.5), (1, 0.73105857863000487)])
def test_sigmoid(input, output):
    np.testing.assert_almost_equal(utils.sigmoid(input), output)

def test_tonp():
    tensor = torch.tensor([1, 2, 3], dtype=torch.int16)
    assert np.all(utils.tonp(tensor) == np.array([1, 2, 3], dtype=np.int16))

def test_tonp_list():
    tensors = [
        torch.tensor([1, 2, 3]), 
        torch.tensor([4, 5, 6])
    ]

    arrs = utils.tonp(tensors)
    assert np.all(arrs[0] == np.array([1, 2, 3]))
    assert np.all(arrs[1] == np.array([4, 5, 6]))

def test_totensor():
    arr = np.array([0.5, 1, 1.5])

    tensor = utils.totensor(arr, device='cpu', type='float32')
    assert torch.all(tensor == torch.tensor([0.5, 1, 1.5], dtype=torch.float32))
    assert tensor.device == torch.device('cpu')
    assert tensor.dtype == torch.float32


def test_torch_accuracy():
    y = np.array([0, 1, 2, 4])
    outputs = torch.tensor([
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.6, 1.7, 0.8, 0.9, 1.0],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.1, 0.1, 0.3, 0.4, 0.5]
    ])

    print(outputs.shape)
    print(outputs.argmax(1))

    assert utils.torch_accuracy(y, outputs) == 0.5