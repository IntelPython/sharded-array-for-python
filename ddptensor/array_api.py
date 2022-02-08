creators = [
    "arange",  # (start, /, stop=None, step=1, *, dtype=None, device=None)
    "asarray",  # (obj, /, *, dtype=None, device=None, copy=None)
    "empty",
    "empty_like",  # (x, /, *, dtype=None, device=None)
    "eye",  # (n_rows, n_cols=None, /, *, k=0, dtype=None, device=None)
    "from_dlpack",  # (x, /)
    "full",
    "full_like",  # (x, /, fill_value, *, dtype=None, device=None)
    "linspace",  # (start, stop, /, num, *, dtype=None, device=None, endpoint=True)
    "meshgrid",  # (*arrays, indexing=’xy’)
    "ones",
    "ones_like",  # (x, /, *, dtype=None, device=None)
    "zeros",
    "zeros_like",  # (x, /, *, dtype=None, device=None)
]

ew_binary_methods_inplace = [
    # inplace operators
    "__iadd__",
    "__iand__",
    "__ifloordiv__",
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
    "__rfloordiv__",
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
