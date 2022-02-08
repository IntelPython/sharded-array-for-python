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
