from . import _ddptensor as _cdt
from . import dtensor


def get_slice(obj, *args):
    return _cdt._get_slice(obj._t, *args)


def get_locals(obj):
    return _cdt._get_locals(obj._t, obj)


def from_locals(objs):
    arg = objs if isinstance(objs, (list, tuple)) else [objs]
    return dtensor(_cdt._from_locals(arg))


def gather(obj, root=_cdt._Ranks._REPLICATED):
    return _cdt._gather(obj._t, root)
