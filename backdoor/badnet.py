from typing import Callable, Optional, Tuple
import numpy as np
import torch

from backdoor.utils import tonp, totensor

from . import image_utils

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

    def apply(self, data: image_utils.AnyImageArray, poison_only: bool=False, sample_weight: Optional[float]=None) -> image_utils.ScikitImageArray:
        """
        Apply the BadNets attack on some input data.
        The input X can be in scikit or torch format. The resultant samples are returned in scikit format.
        """
        X, y = data

        # normalize to scikit format, which allows for RGBA compositing
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

        if sample_weight is not None:
            weights = [1.]*len(X) + [sample_weight]*len(extra_X)
            return retx, rety, np.array(weights)
        else:
            return retx, rety

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

        return newX, newy