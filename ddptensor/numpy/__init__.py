from .. import empty, float32


def fromfunction(function, shape, *, dtype=float32, team=1):
    t = empty(shape, dtype, team)
    t._t.map(function)
    return t
