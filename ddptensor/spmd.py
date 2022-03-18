from . import _ddptensor as _cdt

def get_slice(obj, *args):
    return _cdt._get_slice(obj._t, *args)

def get_local(obj):
    return  _cdt._get_local(obj._t, obj)

def gather(obj, root=_cdt._Ranks._REPLICATED):
    return  _cdt._gather(obj._t, root)
