"""
The Tensor class for ddptensor, a
distributed implementation of the array API as defined here:
https://data-apis.org/array-api/latest
"""
#
# See __init__.py for an implementation overview
#
from . import _ddptensor as _cdt
from . import array_api as api


def slicefy(x):
    if isinstance(x, slice):
        return x
    # slice that extracts a single element at index x
    next_val = None if x == -1 else x + 1
    return slice(x, next_val, 1)


class dtensor:
    def __init__(self, t):
        self._t = t

    def __repr__(self):
        return self._t.__repr__()

    for method in api.api_categories["EWBinOp"]:
        if method.startswith("__"):
            METHOD = method.upper()
            exec(
                f"{method} = lambda self, other: dtensor(_cdt.EWBinOp.op(_cdt.{METHOD}, self._t, other._t if isinstance(other, dtensor) else other))"
            )

    # inplace operators still lead to an assignment, needs more involved analysis
    # for method in api.api_categories["IEWBinOp"]:
    #     METHOD = method.upper()
    #     exec(
    #         f"{method} = lambda self, other: (self, _cdt.IEWBinOp.op(_cdt.{METHOD}, self._t, other._t if isinstance(other, dtensor) else other)[0])"
    #     )

    for method in api.api_categories["EWUnyOp"]:
        if method.startswith("__"):
            METHOD = method.upper()
            exec(
                f"{method} = lambda self: dtensor(_cdt.EWUnyOp.op(_cdt.{METHOD}, self._t))"
            )

    for method in api.api_categories["UnyOp"]:
        exec(f"{method} = lambda self: self._t.{method}()")

    for att in api.attributes:
        exec(f"{att} = property(lambda self: self._t.{att})")

    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        key = [slicefy(x) for x in key]
        return dtensor(self._t.__getitem__(key))

    def __setitem__(self, key, value):
        key = key if isinstance(key, tuple) else (key,)
        key = [slicefy(x) for x in key]
        self._t.__setitem__(key, value._t if isinstance(value, dtensor) else value)
