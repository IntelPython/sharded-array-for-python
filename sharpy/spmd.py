from . import _sharpy as _csp
from . import ndarray


def get_slice(obj, *args):
    return _csp._get_slice(obj._t, *args)


def get_locals(obj):
    return _csp._get_locals(obj._t, obj)


def from_locals(objs):
    arg = objs if isinstance(objs, (list, tuple)) else [objs]
    return ndarray(_csp._from_locals(arg))


def gather(obj, root=_csp._Ranks._REPLICATED):
    return _csp._gather(obj._t, root)
