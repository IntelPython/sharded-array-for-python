from . import _ddptensor as _cdt
from ._ddptensor import float64, int64, fini
from . import array_api as api

unary_methods = [
    # "__array_namespace__",  # (self, /, *, api_version=None)
    "__bool__",  # (self, /)
    # "__dlpack__",  # (self, /, *, stream=None)
    # "__dlpack_device__",  # (self, /)
    "__float__",  # (self, /)
    "__int__",  # (self, /)
    "__len__",  # (self, /)
]

t_attributes = ["dtype", "shape", ]  #"device", "ndim", "size", "T"]

#def try_except(func, *args, **kwargs):
#    try:
#        return func(*args, **kwargs)
#    except:
#        return None
    
class dtensor:
    def __init__(self, t):
        self._t = t

    def __repr__(self):
        return self._t.__repr__()


    for method in api.ew_binary_methods:
        METHOD = method.upper()
        exec(
            f"{method} = lambda self, other: dtensor(_cdt.EWBinOp.op(_cdt.{METHOD}, self._t, other._t))" # if isinstance(other, dtensor) else other, True))"
        )

    for method in api.ew_binary_methods_inplace:
        METHOD = method.upper()
        exec(
            f"{method} = lambda self, other: (self, _cdt.IEWBinOp.op(_cdt.{METHOD}, self._t, other._t))[0]" # if isinstance(other, dtensor) else other))[0]"
        )

    for method in api.ew_unary_methods:
        METHOD = method.upper()
        exec(
            f"{method} = lambda self: dtensor(_cdt.EWUnyOp.op(_cdt.{METHOD}, self._t))"
        )
        
    # for method in unary_methods:
    #     exec(
    #         f"{method} = lambda self: self._t.{method}()"
    #     )

    # for att in t_attributes:
    #     exec(
    #         f"{att} = property(lambda self: self._t.{att})"
    #     )

    # def __getitem__(self, *args):
    #     x = self._t.__getitem__(*args)
    #     return dtensor(x)

    # def __setitem__(self, key, value):
    #     x = self._t.__setitem__(key, value._t if isinstance(value, dtensor) else value)

    # def get_slice(self, *args):
    #     return self._t.get_slice(*args)
