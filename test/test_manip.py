import itertools
import os

import numpy
import pytest
from mpi4py import MPI
from utils import device, runAndCompare

import sharpy as sp


class TestManip:
    def test_reshape1(self):
        def doit(aapi, **kwargs):
            a = aapi.arange(0, 12 * 11, 1, aapi.int32, **kwargs)
            return aapi.reshape(a, [6, 22])

        assert runAndCompare(doit)

    def test_reshape2(self):
        def doit(aapi, **kwargs):
            a = aapi.arange(0, 12 * 11, 1, aapi.int32, **kwargs)
            b = aapi.reshape(a, [12, 11])
            c = b[0:12:2, 0:10:2]
            return aapi.reshape(c, [5, 6])

        assert runAndCompare(doit)

    @pytest.mark.skipif(len(device), reason="FIXME 64bit on GPU")
    def test_astype_f64i32(self):
        def doit(aapi, **kwargs):
            a = aapi.arange(0, 8, 1, aapi.float64, **kwargs)
            a += 0.3
            return a.astype(aapi.int32)

        assert runAndCompare(doit)

    def test_astype_view(self):
        a = sp.arange(0, 8, 1, sp.int32, device=device)
        b = a.astype(sp.int32)
        b[:3] = 5
        assert b.dtype == sp.int32
        assert numpy.allclose(sp.to_numpy(a), [5, 5, 5, 3, 4, 5, 6, 7])

    @pytest.mark.skipif(
        len(device)
        or MPI.COMM_WORLD.size > 1
        or os.getenv("SHARPY_FORCE_DIST") is not None,
        reason="FIXME GPU and multi-proc",
    )
    def test_astype_copy(self):
        a = sp.arange(0, 8, 1, sp.int32, device=device)
        b = a.astype(sp.int32, copy=True)
        b[:3] = 5
        assert b.dtype == sp.int32
        assert numpy.allclose(sp.to_numpy(a), [0, 1, 2, 3, 4, 5, 6, 7])
        assert numpy.allclose(sp.to_numpy(b), [5, 5, 5, 3, 4, 5, 6, 7])

    def test_astype_suite(self):
        dtype_list = [
            sp.int32,
            sp.uint32,
            sp.float32,
        ]
        if len(device) == 0:
            dtype_list += [
                sp.int64,
                sp.uint64,
                sp.float64,
            ]

        for from_type, to_type in itertools.product(dtype_list, dtype_list):
            a = sp.arange(0, 8, 1, dtype=from_type, device=device)
            b = a.astype(to_type)
            assert b.dtype == to_type
            assert numpy.allclose(sp.to_numpy(b), [0, 1, 2, 3, 4, 5, 6, 7])

    @pytest.mark.skipif(len(device), reason="n.a.")
    def test_todevice_host2host(self):
        a = sp.arange(0, 8, 1, sp.int32)
        b = a.to_device()
        assert numpy.allclose(sp.to_numpy(b), [0, 1, 2, 3, 4, 5, 6, 7])

    @pytest.mark.skip(reason="mixed CPU/GPU support")
    def test_todevice_host2gpu(self):
        a = sp.arange(0, 8, 1, sp.int32)
        b = a.to_device(device="GPU")
        assert numpy.allclose(sp.to_numpy(b), [0, 1, 2, 3, 4, 5, 6, 7])
