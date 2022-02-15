from . import _ddptensor as _cdt
from . ddptensor import dtensor

def uniform(low, high, size, dtype=_cdt.float64):
    return dtensor(_cdt.Random.uniform(dtype, size, low, high))

def seed(s):
    _cdt.Random.seed(s)
