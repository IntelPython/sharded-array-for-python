import ndarray

from .. import _sharpy as _csp


def manual_seed(s=None):
    ndarray.__impl.manual_seed(s + _csp.myrank() if s else _csp.myrank())


def _rand_torch(shape, *args, **kwargs):
    return ndarray.__impl.rand(tuple(shape))


def rand(shape, *args, **kwargs):
    return ndarray.ndarray(
        _csp.create(shape, "_rand_torch", "ndarray.torch", *args, **kwargs)
    )


def erf(ary):
    return ndarray.ndarray(_csp.ew_unary_op(ary._t, "erf", False))
