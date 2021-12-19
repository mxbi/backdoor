from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np
from scipy import stats

class Parameter:
    def sample(self):
        """
        Randomly sample this parameter
        """
        raise NotImplementedError

@dataclass
class Uniform(Parameter):
    """
    A uniformly-distributed parameter, in the range [start, end).

    If `integer`, then returns a integer sample in the range [start, end)
    """
    start: float
    end: float
    integer: bool = False

    def sample(self):
        if self.integer:
            return np.random.randint(self.start, self.end)
        else:
            return np.random.random() * (self.end - self.start) + self.start

@dataclass
class LogUniform(Parameter):
    """
    A log-uniformly distributed parameter, in the range [start, end)
    The samples will be uniformly distributed in order of magnitude.
    
    If `integer`, then an integer value will be returned (by flooring the uniform value)
    """
    start: float
    end: float
    integer: bool = False

    def __post_init__(self):
        self.rv = stats.loguniform(self.start, self.end)

    def sample(self):
        samp = self.rv.rvs()
        if self.integer:
            samp = int(samp)
        return samp

@dataclass
class Boolean(Parameter):
    """
    A boolean parameter. By default, p=0.5
    """
    p: float = 0.5

    def sample(self):
        return np.random.random() < self.p

@dataclass
class Choice(Parameter):
    """
    Samples randomly from the given choices
    """
    choices: List[Any]

    def sample(self):
        return np.random.choice(self.choices)

class Searchable:
    def __init__(self, func: Callable):
        """
        Takes a function and wraps it such that the function can now take Parameters as arguments.
        """
        self.func = func

    def _resolve_arg(self, arg):
        if isinstance(arg, Parameter):
            return arg.sample()
        else:
            return arg

    def __call__(self, *args, **kwargs):
        new_args = [self._resolve_arg(arg) for arg in args]
        new_kwargs = {k: self._resolve_arg(v) for k, v in kwargs.items()}

        return self.func(*new_args, **new_kwargs)

    def with_args(self, *args, **kwargs):
        """
        Call the searchable function, and return the arguments that were _actually_ used.

        Returns: args, kwargs, func(args, kwargs)
        """

        new_args = [self._resolve_arg(arg) for arg in args]
        new_kwargs = {k: self._resolve_arg(v) for k, v in kwargs.items()}

        return new_args, new_kwargs, self.func(*new_args, **new_kwargs)        

    def random_search(self, args, kwargs, trials, pbar=True, seed=42, return_args=True, on_error='raise'):
        """
        trials: The number of times to execute the function
        pbar: Whether to display a tqdm progress bar
        seed: The random seed to use
        return_args: When returning the result, also return the arguments and kwargs that were used in each call
        on_error: One of ['raise', 'return']. If 'return', the exception will be returned in the results list instead of an actual result. KeyboardInterrupt is always re-raised.
        """
        assert on_error in ['raise', 'return']

        res = []

        # Optional progressbar
        loop = range(trials)
        if pbar:
            from tqdm import tqdm
            loop = tqdm(loop)

        if return_args:
            callable = self.with_args
        else:
            callable = self

        for i in loop:
            if on_error == 'return':
                # Catch all errors except KeyboardInterrupt (to allow the user to exit)
                try:
                    res.append(callable(args, kwargs))
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    res.append(e)
            else:
                res.append(callable(args, kwargs))

        return res
