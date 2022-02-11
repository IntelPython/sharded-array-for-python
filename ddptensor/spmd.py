from . import _ddptensor as _cdt

def get_slice(self, *args):
    return _cdt._get_slice(self._t, *args)
