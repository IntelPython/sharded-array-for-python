from . import _ddptensor as _cdt
from . import float64
from . ddptensor import dtensor

def uniform(low, high, size, dtype=float64):
    return dtensor(_cdt.Random.uniform(dtype, size, low, high))

def seed(s):
    _cdt.Random.seed(s)
