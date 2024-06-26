from .. import __impl
from .. import _sharpy as _csp
from ..ndarray import ndarray


def seed(s=None):
    __impl.random.seed(s + _csp.myrank() if s else _csp.myrank())


def _uniform_numpy(shape, start, stop, dtype=None):
    return __impl.random.uniform(start, stop, shape)


def uniform(start, stop, shape):
    return ndarray(
        _csp.create(
            shape, "_uniform_numpy", "ndarray.numpy.random", start, stop
        )
    )
