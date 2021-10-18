class BadNetDataPoisoning:
    """
    This class sets up a data poisoning attack in the style of the BadNets paper.
    
    You provide a poisoning_func(x, y) which takes in a single sample, and returns None if this example is not of interest of attack.
    If the function returns a single poisoned (x, y) pair in return, this data point will be appended to the end of the dataset.
    """
    def __init__(self, poisoning_func):
        self.poisoning_func = poisoning_func

    @classmethod
    def pattern_backdoor(self, old_class, new_class, pattern):
        def poisoning_func(xsamp, ysamp):
            if ysamp == old_class:
                poisoned_xsamp = xsamp + pattern # TODO: fix this shit
                return poisoned_xsamp, new_class
        return self(poisoning_func)

    def apply(self, data, poison_only=False):
        X, y = data
        
        extra_X = []
        extra_y = []
        for xsamp, ysamp in zip(X, y):
            if p := self.poisoning_func(xsamp, ysamp):
                extra_X.append(p[0])
                extra_y.append(p[1])

        return np.concatenate([X, np.array(extra_X)]), np.concatenate([y, np.array(extra_Y)])
