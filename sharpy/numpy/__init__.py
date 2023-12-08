from .. import empty, float32


def fromfunction(function, shape, *, dtype=float32, device="", team=1):
    t = empty(shape, dtype=dtype, device=device, team=team)
    t._t.map(function)
    return t
