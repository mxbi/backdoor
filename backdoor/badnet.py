import numpy as np

from . import image_utils

class BadNetDataPoisoning:
    """
    This class sets up a data poisoning attack in the style of the BadNets paper.
    
    You provide a poisoning_func(x, y) which takes in a single sample, and returns None if this example is not of interest of attack.
    If the function returns a single poisoned (x, y) pair in return, this data point will be appended to the end of the dataset.
    """
    def __init__(self, poisoning_func):
        self.poisoning_func = poisoning_func

    @classmethod
    def pattern_backdoor(self, orig_class, backdoor_class, patch):
        """
        Setup a BadNets attack with the following property:

        For any x such that f(x)=orig_class, f([x+patch])=backdoor_class.
        The patch is applied as an RGBA filter using image_utils.overlay_transparent_patch
        """
        def poisoning_func(xsamp, ysamp):
            if ysamp == orig_class or orig_class is None:
                # poisoned_xsamp = xsamp + pattern
                poisoned_xsamp = image_utils.overlay_transparent_patch(xsamp, patch)
                return poisoned_xsamp, backdoor_class
        return self(poisoning_func)

    def apply(self, data, poison_only=False):
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

        return np.concatenate([X, np.array(extra_X)]), np.concatenate([y, np.array(extra_y)])
