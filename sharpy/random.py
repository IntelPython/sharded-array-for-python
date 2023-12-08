from . import _sharpy as _csp
from . import float64
from .sharpy import ndarray


def uniform(low, high, size, dtype=float64):
    return ndarray(_csp.Random.uniform(dtype, size, low, high))


def seed(s):
    _csp.Random.seed(s)
