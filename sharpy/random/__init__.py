import numpy as np

import sharpy as sp
from sharpy import float64
from sharpy.numpy import fromfunction


def uniform(low, high, size, device="", team=1):
    data = np.random.uniform(low, high, size)
    if len(data.shape) == 0:
        sp_data = sp.full((), data[()], device=device, team=team)
        return sp_data
    return fromfunction(
        lambda *index: data[index],
        data.shape,
        dtype=float64,
        device=device,
        team=team,
    )


def rand(*shape, device="", team=1):
    data = np.random.rand(*shape)
    if isinstance(data, float):
        return data
    return fromfunction(
        lambda *index: data[index],
        data.shape,
        dtype=float64,
        device=device,
        team=team,
    )


def seed(s):
    np.random.seed(s)
