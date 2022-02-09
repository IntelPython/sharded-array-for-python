from . import _ddptensor as _cdt
from .ddptensor import float64, int64, fini, dtensor
from os import getenv
from . import array_api as api

#__impl_str = getenv("DDPNP_ARRAY", 'numpy')
#exec(f"import {__impl_str} as __impl")

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
