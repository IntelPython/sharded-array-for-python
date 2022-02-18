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

from . import _ddptensor as _cdt
from ._ddptensor import (
    FLOAT64 as float64,
    FLOAT32 as float32,
    INT64 as int64,
    INT32 as int32,
    INT16  as int16,
    INT8 as int8,
    UINT64 as uint64,
    UINT32 as uint32,
    UINT16 as uint16,
    UINT8 as uint8,
    fini
)
from .ddptensor import dtensor
from os import getenv
from . import array_api as api
from . import spmd

for op in api.api_categories["EWBinOp"]:
    if not op.startswith("__"):
        OP = op.upper()
        exec(
            f"{op} = lambda this, other: dtensor(_cdt.EWBinOp.op(_cdt.{OP}, this._t, other._t if isinstance(other, ddptensor) else other))"
        )

for op in api.api_categories["EWUnyOp"]:
    if not op.startswith("__"):
        OP = op.upper()
        exec(
            f"{op} = lambda this: dtensor(_cdt.EWUnyOp.op(_cdt.{OP}, this._t))"
        )

for func in api.api_categories["Creator"]:
    FUNC = func.upper()
    if func in ["empty", "ones", "zeros",]:
        exec(
            f"{func} = lambda shape, dtype: dtensor(_cdt.Creator.create_from_shape(_cdt.{FUNC}, shape, dtype))"
        )
    elif func == "full":
        exec(
            f"{func} = lambda shape, val, dtype: dtensor(_cdt.Creator.full(_cdt.{FUNC}, shape, val, dtype))"
        )

for func in api.api_categories["ReduceOp"]:
    FUNC = func.upper()
    exec(
        f"{func} = lambda this, dim: dtensor(_cdt.ReduceOp.op(_cdt.{FUNC}, this._t, dim))"
    )
