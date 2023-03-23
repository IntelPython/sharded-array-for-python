from .. import empty, float32


def fromfunction(function, shape, *, dtype=float32):
    t = empty(shape, dtype)
    t._t.map(function)
    return t
