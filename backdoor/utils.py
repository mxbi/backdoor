import numpy as np

# From mlcrate
# We lazy-load torch the first time one of the functions that requires it is called
torch = None

def _check_torch_import():
    global torch
    if torch is not None:
        return
    import importlib
    torch = importlib.import_module('torch')

def sigmoid(x):
    x = tonp(x)
    return 1 / (1 + np.exp(-x))

def tonp(tensor):
    """Takes any PyTorch tensor and converts it to a numpy array or scalar as appropiate.
    When given something that isn't a PyTorch tensor, it will attempt to convert to a NumPy array or scalar anyway.
    Not heavily optimized.
    
    If you provide a list of tensors, the return result will be a list of numpy arrays, NOT a single numpy array."""
    _check_torch_import()
    if isinstance(tensor, torch.Tensor):
        arr = tensor.data.detach().cpu().numpy()
    
    elif isinstance(tensor, list):
        return [tonp(t) for t in tensor]
    elif isinstance(tensor, tuple):
        return (tonp(t) for t in tensor)

    else: # It's not a tensor! We'll handle it anyway
        arr = np.array(tensor)
    if arr.shape == ():
        return np.asscalar(arr)
    else:
        return arr

def totensor(arr, device=None, type='float32'):
    """Converts any array-like or scalar to a PyTorch tensor, and checks that the array is in the correct type (defaults to float32) and on the correct device.
    Equivalent to calling `torch.from_array(np.array(arr, dtype=type)).to(device)` but more efficient.
    NOTE: If the input is a torch tensor, the type will not be checked.
    Keyword arguments:
    arr -- Any array-like object (eg numpy array, list, numpy varaible)
    device (optional) -- Move the tensor to this device after creation
    type -- the numpy data type of the tensor. Defaults to 'float32' (regardless of the input)
    Returns:
    tensor - A torch tensor"""
    _check_torch_import()
    # If we're given a tensor, send it right back.
    if isinstance(arr, torch.Tensor):
        if device:
            return arr.to(device) # If tensor is already on the specified device, this doesn't copy the tensor.
        else:
            return arr

    # Only call np.array() if it is not already an array, otherwise numpy will waste time copying the array
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    # Likewise with type conversion
    if arr.dtype != type:
        arr = arr.astype(type, copy=False)

    tensor = torch.from_numpy(arr)
    if device:
        tensor = tensor.to(device)
    return tensor

def torch_accuracy(y, outputs):
    """
    Returns the accuracy [0,1] of the 2-D prediction array `outputs` against the 1-D int array `y`. Prediction is taken as the argmax of `outputs`.
    """
    outputs_cpu = tonp(outputs)
    acc = (y == outputs_cpu.argmax(1)).mean()
    return acc