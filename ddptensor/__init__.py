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
from .ddptensor import float64, int64, fini, dtensor
from os import getenv
from . import array_api as api
from . import spmd

for op in api.ew_binary_ops:
    OP = op.upper()
    exec(
        f"{op} = lambda this, other: dtensor(_cdt.EWBinOp.op(_cdt.{OP}, this._t, other._t))"  # if isinstance(other, ddptensor) else other, False))"
    )

for op in api.ew_unary_ops:
    OP = op.upper()
    exec(
        f"{op} = lambda this: dtensor(_cdt.EWUnyOp.op(_cdt.{OP}, this._t))"
    )

for func in api.creators:
    FUNC = func.upper()
    if func in ["empty", "ones", "zeros",]:
        exec(
            f"{func} = lambda shape, dtype: dtensor(_cdt.Creator.create_from_shape(_cdt.{FUNC}, shape, dtype))"
        )
    elif func == "full":
        exec(
            f"{func} = lambda shape, val, dtype: dtensor(_cdt.Creator.full(_cdt.{FUNC}, shape, val, dtype))"
        )

for func in api.statisticals:
    FUNC = func.upper()
    exec(
        f"{func} = lambda this, dim: dtensor(_cdt.ReduceOp.op(_cdt.{FUNC}, this._t, dim))"
    )
