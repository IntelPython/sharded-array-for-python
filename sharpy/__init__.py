"""
Distributed implementation of the array API as defined here:
https://data-apis.org/array-api/latest
"""

# Many features of the API are very uniformly defined.
# We make use of that by providing lists of operations which are similar
# (see array_api.py). __init__.py and sharpy.py simply generate the API
# by iterating through these lists and forwarding the function calls the the
# C++-extension. Python functions are defined and added by using "eval".
# For many operations we assume the C++-extension defines enums which allow
# us identifying each operation.
# At this point there are no checks of input arguments whatsoever, arguments
# are simply forwarded as-is.

import os
import re
from importlib import import_module
from os import getenv
from typing import Any

from . import _sharpy as _csp
from . import array_api as api
from . import spmd
from ._sharpy import BOOL as _bool
from ._sharpy import FLOAT32 as float32
from ._sharpy import FLOAT64 as float64
from ._sharpy import INT8 as int8
from ._sharpy import INT16 as int16
from ._sharpy import INT32 as int32
from ._sharpy import INT64 as int64
from ._sharpy import UINT8 as uint8
from ._sharpy import UINT16 as uint16
from ._sharpy import UINT32 as uint32
from ._sharpy import UINT64 as uint64
from ._sharpy import fini
from ._sharpy import init as _init
from ._sharpy import sync
from .ndarray import ndarray


# Lazy load submodules
def __getattr__(name):
    if name == "random":
        import sharpy.random as random

        return random
    elif name == "numpy":
        import sharpy.numpy as numpy

        return numpy

    if "_fallback" in globals():
        return _fallback(name)


_sharpy_cw = bool(int(getenv("SHARPY_CW", False)))

pi = 3.1415926535897932384626433


def init(cw=None):
    libidtr = os.path.join(os.path.dirname(__file__), "libidtr.so")
    assert os.path.isfile(libidtr), "libidtr.so not found"

    cw = _sharpy_cw if cw is None else cw
    _init(cw, libidtr)


def to_numpy(a):
    return _csp.to_numpy(a._t)


for op in api.api_categories["EWBinOp"]:
    if not op.startswith("__"):
        OP = op.upper()
        exec(
            f"{op} = lambda this, other: ndarray(_csp.EWBinOp.op(_csp.{OP}, this._t if isinstance(this, ndarray) else this, other._t if isinstance(other, ndarray) else other))"
        )

for op in api.api_categories["EWUnyOp"]:
    if not op.startswith("__"):
        OP = op.upper()
        exec(
            f"{op} = lambda this: ndarray(_csp.EWUnyOp.op(_csp.{OP}, this._t))"
        )


def _validate_device(device):
    if len(device) == 0 or re.search(
        r"^((opencl|level-zero|cuda):)?(host|gpu|cpu|accelerator)(:\d+)?$",
        device,
    ):
        return device
    else:
        raise ValueError(f"Invalid device string: {device}")


def arange(start, /, end=None, step=1, dtype=int64, device="", team=1):
    if end is None:
        end = start
        start = 0
    assert step != 0, "step cannot be zero"
    if (end - start) * step < 0:
        # invalid range, return empty array
        start = end = 0
        step = 1
    return ndarray(
        _csp.Creator.arange(
            start, end, step, dtype, _validate_device(device), team
        )
    )


for func in api.api_categories["Creator"]:
    FUNC = func.upper()
    if func == "full":
        exec(
            f"{func} = lambda shape, val, dtype=float64, device='', team=1: ndarray(_csp.Creator.full(shape, val, dtype, _validate_device(device), team))"
        )
    elif func == "empty":
        exec(
            f"{func} = lambda shape, dtype=float64, device='', team=1: ndarray(_csp.Creator.full(shape, None, dtype, _validate_device(device), team))"
        )
    elif func == "ones":
        exec(
            f"{func} = lambda shape, dtype=float64, device='', team=1: ndarray(_csp.Creator.full(shape, 1, dtype, _validate_device(device), team))"
        )
    elif func == "zeros":
        exec(
            f"{func} = lambda shape, dtype=float64, device='', team=1: ndarray(_csp.Creator.full(shape, 0, dtype, _validate_device(device), team))"
        )
    elif func == "linspace":
        exec(
            f"{func} = lambda start, end, step, endpoint, dtype=float64, device='', team=1: ndarray(_csp.Creator.linspace(start, end, step, endpoint, dtype, _validate_device(device), team))"
        )


for func in api.api_categories["ManipOp"]:
    FUNC = func.upper()
    if func == "reshape":
        exec(
            f"{func} = lambda this, shape, cp=None: ndarray(_csp.ManipOp.reshape(this._t, shape, cp))"
        )
    elif func == "permute_dims":
        exec(
            f"{func} = lambda this, axes: ndarray(_csp.ManipOp.permute_dims(this._t, axes))"
        )

for func in api.api_categories["ReduceOp"]:
    FUNC = func.upper()
    exec(
        f"{func} = lambda this, dim=None: ndarray(_csp.ReduceOp.op(_csp.{FUNC}, this._t, dim if dim else []))"
    )

for func in api.api_categories["LinAlgOp"]:
    FUNC = func.upper()
    if func in [
        "tensordot",
        "vecdot",
    ]:
        exec(
            f"{func} = lambda this, other, axis: ndarray(_csp.LinAlgOp.{func}(this._t, other._t, axis))"
        )
    elif func == "matmul":
        exec(
            f"{func} = lambda this, other: ndarray(_csp.LinAlgOp.vecdot(this._t, other._t, 0))"
        )
    elif func == "matrix_transpose":
        exec(f"{func} = lambda this: ndarray(_csp.LinAlgOp.{func}(this._t))")


_fb_env = getenv("SHARPY_FALLBACK")
if _fb_env is not None:
    if not _fb_env.isalnum():
        raise ValueError(f"Invalid SHARPY_FALLBACK value '{_fb_env}'")

    class _fallback:
        "Fallback to whatever is provided in SHARPY_FALLBACK"
        try:
            _fb_lib = import_module(_fb_env)
        except ModuleNotFoundError:
            raise ValueError(
                f"Invalid SHARPY_FALLBACK value '{_fb_env}': module not found"
            )

        def __init__(self, fname: str, mod=None) -> None:
            """get callable with name 'fname' from fallback-lib
            or throw exception"""
            self._mod = mod if mod else _fallback._fb_lib
            self._func = getattr(self._mod, fname)

        def __call__(self, *args: Any, **kwds: Any) -> Any:
            """convert ndarrays args to fallback arrays,
            call fallback-lib and return converted ndarray"""
            nargs = []
            nkwds = {}
            for arg in args:
                nargs.append(
                    spmd.get_locals(arg)[0] if isinstance(arg, ndarray) else arg
                )
            for k, v in kwds.items():
                nkwds[k] = (
                    spmd.get_locals(v)[0] if isinstance(v, ndarray) else v
                )

            res = self._func(*nargs, **nkwds)
            return (
                spmd.from_locals(res)
                if isinstance(res, _fallback._fb_lib.ndarray)
                else res
            )

        def __getattr__(self, name):
            """Attempt to find a fallback in current fallback object.
            This might be necessary if we call something like
            dt.linalg.norm(...)
            """
            return _fallback(name, self._func)
