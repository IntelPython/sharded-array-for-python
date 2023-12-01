"""
Distributed implementation of the array API as defined here:
https://data-apis.org/array-api/latest
"""

# Many features of the API are very uniformly defined.
# We make use of that by providing lists of operations which are similar
# (see array_api.py). __init__.py and ddptensor.py simply generate the API
# by iterating through these lists and forwarding the function calls the the
# C++-extension. Python functions are defined and added by using "eval".
# For many operations we assume the C++-extension defines enums which allow
# us identifying each operation.
# At this point there are no checks of input arguments whatsoever, arguments
# are simply forwarded as-is.

_bool = bool
from typing import Any
from . import _ddptensor as _cdt
from ._ddptensor import (
    FLOAT64 as float64,
    FLOAT32 as float32,
    INT64 as int64,
    INT32 as int32,
    INT16 as int16,
    INT8 as int8,
    UINT64 as uint64,
    UINT32 as uint32,
    UINT16 as uint16,
    UINT8 as uint8,
    BOOL as bool,
    init as _init,
    fini,
    sync,
)

from .ddptensor import dtensor
from os import getenv
from importlib import import_module
from . import array_api as api
from . import spmd

_ddpt_cw = _bool(int(getenv("DDPT_CW", False)))

pi = 3.1415926535897932384626433


def init(cw=None):
    cw = _ddpt_cw if cw is None else cw
    _init(cw)


def to_numpy(a):
    return _cdt.to_numpy(a._t)


for op in api.api_categories["EWBinOp"]:
    if not op.startswith("__"):
        OP = op.upper()
        exec(
            f"{op} = lambda this, other: dtensor(_cdt.EWBinOp.op(_cdt.{OP}, this._t if isinstance(this, ddptensor) else this, other._t if isinstance(other, ddptensor) else other))"
        )

for op in api.api_categories["EWUnyOp"]:
    if not op.startswith("__"):
        OP = op.upper()
        exec(f"{op} = lambda this: dtensor(_cdt.EWUnyOp.op(_cdt.{OP}, this._t))")

for func in api.api_categories["Creator"]:
    FUNC = func.upper()
    if func == "full":
        exec(
            f"{func} = lambda shape, val, dtype=float64, device='', team=1: dtensor(_cdt.Creator.full(shape, val, dtype, device, team))"
        )
    elif func == "empty":
        exec(
            f"{func} = lambda shape, dtype=float64, device='', team=1: dtensor(_cdt.Creator.full(shape, None, dtype, device, team))"
        )
    elif func == "ones":
        exec(
            f"{func} = lambda shape, dtype=float64, device='', team=1: dtensor(_cdt.Creator.full(shape, 1, dtype, device, team))"
        )
    elif func == "zeros":
        exec(
            f"{func} = lambda shape, dtype=float64, device='', team=1: dtensor(_cdt.Creator.full(shape, 0, dtype, device, team))"
        )
    elif func == "arange":
        exec(
            f"{func} = lambda start, end, step, dtype=int64, device='', team=1: dtensor(_cdt.Creator.arange(start, end, step, dtype, device, team))"
        )
    elif func == "linspace":
        exec(
            f"{func} = lambda start, end, step, endpoint, dtype=float64, device='', team=1: dtensor(_cdt.Creator.linspace(start, end, step, endpoint, dtype, device, team))"
        )

for func in api.api_categories["ReduceOp"]:
    FUNC = func.upper()
    exec(
        f"{func} = lambda this, dim=None: dtensor(_cdt.ReduceOp.op(_cdt.{FUNC}, this._t, dim if dim else []))"
    )

# for func in api.api_categories["ManipOp"]:
#     FUNC = func.upper()
#     if func == "reshape":
#         exec(
#             f"{func} = lambda this, /, shape, *, copy=None: dtensor(_cdt.ManipOp.reshape(this._t, shape, copy))"
#         )

for func in api.api_categories["LinAlgOp"]:
    FUNC = func.upper()
    if func in [
        "tensordot",
        "vecdot",
    ]:
        exec(
            f"{func} = lambda this, other, axis: dtensor(_cdt.LinAlgOp.{func}(this._t, other._t, axis))"
        )
    elif func == "matmul":
        exec(
            f"{func} = lambda this, other: dtensor(_cdt.LinAlgOp.vecdot(this._t, other._t, 0))"
        )
    elif func == "matrix_transpose":
        exec(f"{func} = lambda this: dtensor(_cdt.LinAlgOp.{func}(this._t))")


_fb_env = getenv("DDPT_FALLBACK")
if _fb_env is not None:

    class _fallback:
        "Fallback to whatever is provided in DDPT_FALLBACK"
        _fb_lib = import_module(_fb_env)

        def __init__(self, fname: str, mod=None) -> None:
            "get callable with name 'fname' from fallback-lib or throw exception"
            self._mod = mod if mod else _fallback._fb_lib
            self._func = getattr(self._mod, fname)

        def __call__(self, *args: Any, **kwds: Any) -> Any:
            "convert ddptensors args to fallback arrays, call fallback-lib and return converted ddptensor"
            nargs = []
            nkwds = {}
            for arg in args:
                nargs.append(
                    spmd.get_locals(arg)[0] if isinstance(arg, dtensor) else arg
                )
            for k, v in kwds.items():
                nkwds[k] = spmd.get_locals(v)[0] if isinstance(v, dtensor) else v

            res = self._func(*nargs, **nkwds)
            return (
                spmd.from_locals(res)
                if isinstance(res, _fallback._fb_lib.ndarray)
                else res
            )

        def __getattr__(self, name):
            """Attempt to find a fallback in current fallback object.
            This might be necessary if we call something like dt.linalg.norm(...)
            """
            return _fallback(name, self._func)

    def __getattr__(name):
        "Attempt to find a fallback in fallback-lib"
        return _fallback(name)
