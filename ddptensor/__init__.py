from . import _ddptensor as _cdt
from .ddptensor import float64, int64, fini, dtensor
from os import getenv
from . import array_api as api

__impl_str = getenv("DDPNP_ARRAY", 'numpy')
exec(f"import {__impl_str} as __impl")

for op in api.ew_binary_ops:
    exec(
        f"{op} = lambda this, other: dtensor(_cdt.ew_binary_op(this._t, '{op}', other._t if isinstance(other, ddptensor) else other, False))"
    )

for op in api.ew_unary_ops:
    exec(
        f"{op} = lambda this: dtensor(_cdt.ew_unary_op(this._t, '{op}', False))"
    )

for func in api.creators:
    if func in ["empty", "full", "ones", "zeros",]:
        exec(
            f"{func} = lambda shape, *args, **kwargs: dtensor(_cdt.create(shape, '{func}', '{__impl_str}', *args, **kwargs))"
        )

statisticals = [
    "max",   # (x, /, *, axis=None, keepdims=False)
    "mean",  # (x, /, *, axis=None, keepdims=False)
    "min",   # (x, /, *, axis=None, keepdims=False)
    "prod",  # (x, /, *, axis=None, keepdims=False)
    "sum",   # (x, /, *, axis=None, keepdims=False)
    "std",   # (x, /, *, axis=None, correction=0.0, keepdims=False)
    "var",   # (x, /, *, axis=None, correction=0.0, keepdims=False)
]

for func in statisticals:
    exec(
        f"{func} = lambda this, **kwargs: dtensor(_cdt.reduce_op(this._t, '{func}', **kwargs))"
    )
