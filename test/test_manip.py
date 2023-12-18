import sharpy as sp
import numpy
from utils import runAndCompare
import pytest
import itertools
from mpi4py import MPI
import os


class TestManip:
    @pytest.mark.skip(reason="FIXME reshape")
    def test_reshape1(self):
        def doit(aapi):
            a = aapi.arange(0, 12 * 11, 1, aapi.int64)
            return aapi.reshape(a, [6, 22])

        assert runAndCompare(doit)

    @pytest.mark.skip(reason="FIXME reshape")
    def test_reshape2(self):
        def doit(aapi):
            a = aapi.arange(0, 12 * 11, 1, aapi.int64)
            b = aapi.reshape(a, [12, 11])
            c = b[0:12:2, 0:10:2]
            return aapi.reshape(c, [5, 6])

        assert runAndCompare(doit)

    def test_astype_f64i32(self):
        def doit(aapi):
            a = aapi.arange(0, 8, 1, aapi.float64)
            a += 0.3
            return a.astype(aapi.int32)

        assert runAndCompare(doit)

    def test_astype_view(self):
        a = sp.arange(0, 8, 1, sp.int32)
        b = a.astype(sp.int32)
        b[:3] = 5
        assert b.dtype == sp.int32
        assert numpy.allclose(sp.to_numpy(a), [5, 5, 5, 3, 4, 5, 6, 7])

    @pytest.mark.skipif(
        MPI.COMM_WORLD.size > 1 or os.getenv("SHARPY_FORCE_DIST"),
        reason="FIXME multi-proc",
    )
    def test_astype_copy(self):
        a = sp.arange(0, 8, 1, sp.int32)
        b = a.astype(sp.int32, copy=True)
        b[:3] = 5
        assert b.dtype == sp.int32
        assert numpy.allclose(sp.to_numpy(a), [0, 1, 2, 3, 4, 5, 6, 7])
        assert numpy.allclose(sp.to_numpy(b), [5, 5, 5, 3, 4, 5, 6, 7])

    def test_astype_suite(self):
        dtype_list = [
            sp.int32,
            sp.int64,
            sp.uint32,
            sp.uint64,
            sp.float32,
            sp.float64,
        ]

        for from_type, to_type in itertools.product(dtype_list, dtype_list):
            a = sp.arange(0, 8, 1, from_type)
            b = a.astype(to_type)
            assert b.dtype == to_type
            assert numpy.allclose(sp.to_numpy(b), [0, 1, 2, 3, 4, 5, 6, 7])
