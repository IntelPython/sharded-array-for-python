import numpy as np
import pytest
from utils import device

import sharpy as sp


@pytest.fixture(params=[(), (6,), (6, 5), (6, 5, 4)])
def shape(request):
    return request.param


@pytest.fixture(params=[0, 42, 66, 890, 1000])
def seed(request):
    return request.param


def test_random_rand(shape, seed):
    sp.random.seed(seed)
    sp_data = sp.random.rand(*shape, device=device)

    np.random.seed(seed)
    np_data = np.random.rand(*shape)

    if isinstance(np_data, float):
        assert isinstance(sp_data, float) and sp_data == np_data
    else:
        assert np.allclose(sp.to_numpy(sp_data), np_data)


@pytest.mark.parametrize("low,high", [(0, 1), (4, 10), (-100, 100)])
def test_random_uniform(low, high, shape, seed):
    sp.random.seed(seed)
    sp_data = sp.random.uniform(low, high, shape, device=device)

    np.random.seed(seed)
    np_data = np.random.uniform(low, high, shape)

    assert np.allclose(sp.to_numpy(sp_data), np_data)
