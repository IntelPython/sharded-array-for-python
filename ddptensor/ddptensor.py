from . import _ddptensor as _cdt
from ._ddptensor import float64, int64, fini

ew_binary_methods = [
    "__add__",  # (self, other, /)
    "__and__",  # (self, other, /)
    "__eq__",  # (self, other, /)
    "__floordiv__",  # (self, other, /)
    "__ge__",  # (self, other, /)
    "__gt__",  # (self, other, /)
    "__le__",  # (self, other, /)
    "__lshift__",  # (self, other, /)
    "__lt__",  # (self, other, /)
    "__matmul__",  # (self, other, /)
    "__mod__",  # (self, other, /)
    "__mul__",  # (self, other, /)
    "__ne__",  # (self, other, /)
    "__or__",  # (self, other, /)
    "__pow__",  # (self, other, /)
    "__rshift__",  # (self, other, /)
    "__sub__",  # (self, other, /)
    "__truediv__",  # (self, other, /)
    "__xor__",  # (self, other, /)
    # reflected operators
    "__radd__",
    "__rand__",
    "__rflowdiv__",
    "__rlshift__",
    "__rmod__",
    "__rmul__",
    "__ror__",
    "__rpow__",
    "__rrshift__",
    "__rsub__",
    "__rtruediv__",
    "__rxor__",
]

ew_binary_methods_inplace = [
    # inplace operators
    "__iadd__",
    "__iand__",
    "__iflowdiv__",
    "__ilshift__",
    "__imod__",
    "__imul__",
    "__ior__",
    "__ipow__",
    "__irshift__",
    "__isub__",
    "__itruediv__",
    "__ixor__",
]

ew_unary_methods = [
    "__abs__",  # (self, /)
    "__invert__",  # (self, /)
    "__neg__",  # (self, /)
    "__pos__",  # (self, /)
]

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


    for method in ew_binary_methods:
        exec(
            f"{method} = lambda self, other: dtensor(_cdt.ew_binary_op(self._t, '{method}', other._t if isinstance(other, dtensor) else other, True))"
        )

    for method in ew_binary_methods_inplace:
        exec(
            f"{method} = lambda self, other: (self, _cdt.ew_binary_op_inplace(self._t, '{method}', other._t if isinstance(other, dtensor) else other))[0]"
        )

    for method in ew_unary_methods:
        exec(
            f"{method} = lambda self: dtensor(_cdt.ew_unary_op(self._t, '{method}', True))"
        )

    for method in unary_methods:
        exec(
            f"{method} = lambda self: self._t.{method}()"
        )

    for att in t_attributes:
        exec(
            f"{att} = property(lambda self: self._t.{att})"
        )

    def __getitem__(self, *args):
        x = self._t.__getitem__(*args)
        return dtensor(x)

    def __setitem__(self, key, value):
        x = self._t.__setitem__(key, value._t if isinstance(value, dtensor) else value)
