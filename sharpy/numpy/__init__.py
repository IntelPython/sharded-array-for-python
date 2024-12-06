from .. import _team, empty, float32


def fromfunction(function, shape, *, dtype=float32, device="", team=_team):
    t = empty(shape, dtype=dtype, device=device, team=team)
    t._t.map(function)
    return t
