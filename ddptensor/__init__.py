from . import _ddptensor as _cdt
from .ddptensor import float64, int64, fini, dtensor
from os import getenv
from . import array_api as api

__impl_str = getenv("DDPNP_ARRAY", 'numpy')
exec(f"import {__impl_str} as __impl")

ew_binary_ops = [
    "add",  # (x1, x2, /)
    "atan2",  # (x1, x2, /)
    "bitwise_and",  # (x1, x2, /)
    "bitwise_left_shift",  # (x1, x2, /)
    "bitwise_or",  # (x1, x2, /)
    "bitwise_right_shift",  # (x1, x2, /)
    "bitwise_xor",  # (x1, x2, /)
    "divide",  # (x1, x2, /)
    "equal",  # (x1, x2, /)
    "floor_divide",  # (x1, x2, /)
    "greater",  # (x1, x2, /)
    "greater_equal",  # (x1, x2, /)
    "less_equal",  # (x1, x2, /)
    "logaddexp",  # (x1, x2)
    "logical_and",  # (x1, x2, /)
    "logical_or",  # (x1, x2, /)
    "logical_xor",  # (x1, x2, /)
    "multiply",  # (x1, x2, /)
    "less",  # (x1, x2, /)
    "not_equal",  # (x1, x2, /)
    "pow",  # (x1, x2, /)
    "remainder",  # (x1, x2, /)
    "subtract",  # (x1, x2, /)
]

for op in ew_binary_ops:
    exec(
        f"{op} = lambda this, other: dtensor(_cdt.ew_binary_op(this._t, '{op}', other._t if isinstance(other, ddptensor) else other, False))"
    )

ew_unary_ops = [
    "abs",  # (x, /)
    "acos",  # (x, /)
    "acosh",  # (x, /)
    "asin",  # (x, /)
    "asinh",  # (x, /)
    "atan",  # (x, /)
    "atanh",  # (x, /)
    "bitwise_invert",  # (x, /)
    "ceil",  # (x, /)
    "cos",  # (x, /)
    "cosh",  # (x, /)
    "exp",  # (x, /)
    "expm1",  # (x, /)
    "floor",  # (x, /)
    "isfinite",  # (x, /)
    "isinf",  # (x, /)
    "isnan",  # (x, /)
    "logical_not",  # (x, /)
    "log",  # (x, /)
    "log1p",  # (x, /)
    "log2",  # (x, /)
    "log10",  # (x, /)
    "negative",  # (x, /)
    "positive",  # (x, /)
    "round",  # (x, /)
    "sign",  # (x, /)
    "sin",  # (x, /)
    "sinh",  # (x, /)
    "square",  # (x, /)
    "sqrt",  # (x, /)
    "tan",  # (x, /)
    "tanh",  # (x, /)
    "trunc",  # (x, /)
]

for op in ew_unary_ops:
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
