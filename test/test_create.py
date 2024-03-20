import numpy
import pytest
from utils import device, dtypeIsInt, mpi_dtypes

import sharpy as sp


@pytest.fixture(params=mpi_dtypes)
def datatype(request):
    return request.param


@pytest.fixture(params=[(), (6,), (6, 5), (6, 5, 4)])
def shape(request):
    return request.param


@pytest.fixture(
    params=[
        (sp.ones, 1.0),
        (sp.zeros, 0.0),
    ],
    ids=["ones", "zeros"],
)
def creator(request):
    return request.param[0], request.param[1]


def test_create_datatypes(creator, datatype):
    shape = (6, 4)
    func, expected_value = creator
    a = func(shape, dtype=datatype, device=device)
    assert tuple(a.shape) == shape
    assert numpy.allclose(sp.to_numpy(a), [expected_value])


def test_create_shapes(creator, shape):
    datatype = sp.int32
    func, expected_value = creator
    a = func(shape, dtype=datatype, device=device)
    assert tuple(a.shape) == shape
    assert numpy.allclose(sp.to_numpy(a), [expected_value])


@pytest.mark.parametrize("expected_value", [5.0])
def test_full_shapes(expected_value, shape):
    datatype = sp.int32
    value = int(expected_value) if dtypeIsInt(datatype) else expected_value
    a = sp.full(shape, value, dtype=datatype, device=device)
    assert tuple(a.shape) == shape
    assert numpy.allclose(sp.to_numpy(a), [expected_value])


def test_create_invalid_shape(creator):
    shape = (6, -5)
    datatype = sp.int32
    func, expected_value = creator
    with pytest.raises(ValueError):
        func(shape, dtype=datatype, device=device)


def test_full_invalid_shape():
    shape = (6, -5)
    value = 5
    datatype = sp.int32
    with pytest.raises(ValueError):
        sp.full(shape, value, dtype=datatype, device=device)


@pytest.mark.parametrize(
    "start,end,step", [(0, 10, -1), (0, -10, 1), (0, 99999999999999999999, 1)]
)
def tests_arange_invalid(start, end, step):
    with pytest.raises(TypeError):
        sp.arange(start, end, step, dtype=sp.int32, device=device)
