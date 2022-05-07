from typing import Callable, Iterable, Optional, Tuple, Union
import numpy as np
import torch

from backdoor.utils import tonp, totensor

from . import image_utils, dataset
from backdoor.image_utils import ScikitImageArray

class BadNetDataPoisoning:
    """
    This class sets up a data poisoning attack in the style of the BadNets paper.
    
    You provide a poisoning_func(x, y) which takes in a single sample, and returns None if this example is not of interest of attack.
    If the function returns a single poisoned (x, y) pair in return, this data point will be appended to the end of the dataset.
    """
    def __init__(self, poisoning_func: Callable[[image_utils.ScikitImageArray, int], Optional[Tuple[image_utils.ScikitImageArray, int]]]):
        self.poisoning_func = poisoning_func

    @classmethod
    def pattern_backdoor(self, orig_class: Optional[int], backdoor_class: int, patch: image_utils.AnyImageArray):
        """
        Setup a BadNets attack with the following property:

        For any x such that f(x)=orig_class, f([x+patch])=backdoor_class.
        The patch is applied as an RGBA filter using image_utils.overlay_transparent_patch

        If orig_class is None, the backdoor will be applied to all samples.
        """
        def poisoning_func(xsamp, ysamp):
            if ysamp == orig_class or orig_class is None:
                # poisoned_xsamp = xsamp + pattern
                poisoned_xsamp = image_utils.overlay_transparent_patch(xsamp, patch)
                return poisoned_xsamp, backdoor_class
        return self(poisoning_func)

    @classmethod
    def always_backdoor(self, poisoning_func: Callable[[ScikitImageArray], ScikitImageArray], backdoor_class: int):
        """
        Construct a data poisoning attack using a poisoning_func(X).
        Unlike the main constructor, the poisoning function here acts on the image only. It is applied to every image and the resultant class is always backdoor_class.
        """
        def _poisoning_func(xsamp, ysamp):
            return poisoning_func(xsamp), backdoor_class

        return self(_poisoning_func)

    def apply(self, data: image_utils.AnyImageArray, poison_only: bool=False, sample_weight: Optional[float]=None) -> image_utils.ScikitImageArray:
        """
        Apply the BadNets attack on some input data.
        The input X can be in scikit or torch format. The resultant samples are returned in the same format as the input.
        """
        X, y = data

        device = None   
        if isinstance(X, torch.Tensor):
            device = X.device
            X = tonp(X)

        # normalize to scikit format, which allows for RGBA compositing
        input_fmt = image_utils.ImageFormat.detect_format(X)
        if input_fmt == 'torch':
            X = image_utils.ImageFormat.scikit(X)
        
        extra_X = []
        extra_y = []
        for xsamp, ysamp in zip(X, y):
            if p := self.poisoning_func(xsamp, ysamp):
                extra_X.append(p[0])
                extra_y.append(p[1])

        if poison_only:
            retx, rety = np.array(extra_X), np.array(extra_y)
        else:
            retx, rety = np.concatenate([X, np.array(extra_X)]), np.concatenate([y, np.array(extra_y)])

        if input_fmt == 'torch':
            newX = image_utils.ImageFormat.torch(newX)
        if device:
            newX = totensor(newX, device=device)

        if sample_weight is not None:
            weights = [1.]*len(X) + [sample_weight]*len(extra_X)
            return dataset.DataTuple((retx, rety)), np.array(weights)
        else:
            return dataset.DataTuple((retx, rety))

    def apply_random_sample(self, data: image_utils.AnyImageArray, poison_proportion: float) -> image_utils.ScikitImageArray:
        """
        Apply the BadNets attack on some input data.
        This method is faithful to the original paper, in that instead of adding additional data, it replaces `poison_proportion`
        of the data with backdoored examples.

        The data is returned in the same format as it is provided, on the same device (if it is a tensor). The processing is done in Scikit format, and providing torch format will incur a conversion.
        """
        X, y = data

        device = None
        if isinstance(X, torch.Tensor):
            device = X.device
            X = tonp(X)
        
        input_fmt = image_utils.ImageFormat.detect_format(X)
        if input_fmt == 'torch':
            X = image_utils.ImageFormat.scikit(X)

        poisoned_X, poisoned_y = self.apply((X, y), poison_only=True)

        assert len(X) == len(poisoned_X), "`apply_random_sample` is only supported when all data is backdoorable. Check the poisoning function."

        newX = []
        newy = []
        for cx, cy, px, py in zip(X, y, poisoned_X, poisoned_y):
            if np.random.uniform() < poison_proportion:
                newX.append(px)
                newy.append(py)
            else:
                newX.append(cx)
                newy.append(cy)

        newX, newy = np.array(newX), np.array(newy)

        if input_fmt == 'torch':
            newX = image_utils.ImageFormat.torch(newX)
        if device:
            newX = totensor(newX, device=device)

        return dataset.DataTuple((newX, newy))

class Trigger:
    """
    A class for generating trigger functions that can be applied to images
    """

    @staticmethod
    def _cvt_location(location: Union[str, Tuple[int, int]], size: Tuple[int, int], padding: Optional[int]=None) -> Tuple[int, int]:
        """
        Takes a location descriptor (either a corner 'topleft', 'topright', 'bottomleft', 'bottomright' or a tuple of (x, y)), and a size tuple (of the trigger). 
        Optionally takes a padding parameter, which is only valid when a corner is specified.
        Returns a coordinate (x, y) for the top-left-most pixel in the trigger. If the location is right or bottom, the coordinates will be provided in negative space.
        """
        if isinstance(location, str):
            if location not in ['topleft', 'topright', 'bottomleft', 'bottomright']:
                raise TypeError(f"location must be one of 'topleft', 'topright', 'bottomleft', 'bottomright', or int coordinates, not {location}")

            # determine y
            if location in ['topleft', 'bottomleft']:
                y = padding or 0
            elif location in ['topright', 'bottomright']:
                y = - size[1] - (padding or 0)

            # determine x
            if location in ['topleft', 'topright']:
                x = padding or 0
            elif location in ['bottomleft', 'bottomright']:
                x = - size[0] - (padding or 0)

            return (x, y)

        else:
            if isinstance(location, tuple) and list(map(type, location)) == [int, int]:
                if padding is not None:
                    raise ValueError(f"padding is only valid when location is a corner, not {location}")

                return location        
            else:
                raise TypeError(f"location must be one of 'topleft', 'topright', 'bottomleft', 'bottomright', or int coordinates, not {location}")

    @staticmethod
    def from_string(trigger_string):
        return eval(f"Trigger.{trigger_string}")

    @staticmethod
    def _multiple_images_wrapper(fun):
        def trigger_wrapped(X):
            if X.ndim == 4:
                return np.array([fun(img) for img in X])
            else:
                return fun(X)
        trigger_wrapped.__name__ = fun.__name__
        return trigger_wrapped

    @staticmethod
    def checkerboard(location: Union[str, Tuple[int, int]], size: Tuple[int, int], padding: Optional[int]=None,
                    n_channels: int=3, colours=(0, 255)) -> Callable[[image_utils.ScikitImageArray], image_utils.ScikitImageArray]:
        """
        A checkerboard trigger pattern
        """
        x, y = Trigger._cvt_location(location, size, padding)

        # for efficiency, we pre-generate a block that we can blit onto the array
        blit = np.zeros((size[0], size[1], n_channels))
        for i in range(size[0]):
            for j in range(size[1]):
                blit[i, j, :] = [colours[(i + j) % 2]] * n_channels

        def checkerboard_trigger(X):
            X = X.copy()

            # Special case for when it wraps around to zero. E.g [-1:0] gives an empty slice, [-1:None] gives a slice of size 1 as we want.
            x2 = x+size[0]
            if x < 0 and x2 == 0:
                x2 = None
            y2 = y+size[1]
            if y < 0 and y2 == 0:
                y2 = None
            X[x:x2, y:y2, :] = blit
            return X

        checkerboard_trigger = Trigger._multiple_images_wrapper(checkerboard_trigger)
        checkerboard_trigger.trigger_string = f"checkerboard({repr(location)}, {repr(size)}, {repr(padding)}, {repr(n_channels)}, {repr(colours)})"

        return checkerboard_trigger

    @staticmethod
    def block(location: Union[str, Tuple[int, int]], size: Tuple[int, int], padding: Optional[int]=None,
                    n_channels: int=3, colour: Tuple[int]=(255, 255, 255)) -> Callable[[image_utils.ScikitImageArray], image_utils.ScikitImageArray]:
        """
        A single-colour block trigger pattern
        """
        x, y = Trigger._cvt_location(location, size, padding)

        if n_channels != len(colour):
            raise ValueError(f"colour must be a tuple of length {n_channels}")

        def block_trigger(X):
            X = X.copy()
            x2 = x+size[0]
            if x < 0 and x2 == 0:
                x2 = None
            y2 = y+size[1]
            if y < 0 and y2 == 0:
                y2 = None
            X[x:x2, y:y2, :] = colour
            return X

        block_trigger = Trigger._multiple_images_wrapper(block_trigger)
        block_trigger.trigger_string = f"block({repr(location)}, {repr(size)}, {repr(padding)}, {repr(n_channels)}, {repr(colour)})"

        return block_trigger