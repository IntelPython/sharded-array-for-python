import numpy
import pytest
from utils import device, mpi_dtypes, runAndCompare

import sharpy as sp

mpi_idtypes = [  # FIXME
    # sp.int64,
    # sp.uint64,
    sp.int32,
    sp.uint32,
    # sp.int8,
    # sp.uint8,
]


class TestEWB:
    def test_add0(self):
        for dtyp in mpi_dtypes:
            a = sp.ones((6, 6), dtype=dtyp, device=device)
            c = a + a
            r1 = sp.sum(c)
            v = 6 * 6 * 2
            assert float(r1) == v

    def test_add1(self):
        for dtyp in mpi_dtypes:
            a = sp.ones((6, 6), dtype=dtyp, device=device)
            b = sp.ones((6, 6), dtype=dtyp, device=device)
            c = a + b
            r1 = sp.sum(c)
            v = 6 * 6 * 2
            assert float(r1) == v

    def test_add2(self):
        for dtyp in mpi_idtypes:
            a = sp.ones((16, 16), dtype=dtyp, device=device)
            c = a + 1
            r1 = sp.sum(c, [0, 1])
            v = 16 * 16 * 2
            assert float(r1) == v

    def test_add3(self):
        for dtyp in mpi_idtypes:
            a = sp.ones((16, 16), dtype=dtyp, device=device)
            b = sp.ones((16, 16), dtype=dtyp, device=device)
            c = a + b + 1
            r1 = sp.sum(c, [0, 1])
            v = 16 * 16 * 3
            assert float(r1) == v

    def test_add4(self):
        for dtyp in mpi_idtypes:
            n = 16
            a = sp.fromfunction(
                lambda i, j: i, (n, n), dtype=dtyp, device=device
            )
            b = sp.ones((n, n), dtype=dtyp, device=device)
            c = a + b
            a[:, :] = a[:, :] + c[:, :]
            r1 = sp.sum(a, [0, 1])
            v = n * n * n
            assert float(r1) == v

    def test_add_mul(self):
        def doit(aapi, **kwargs):
            a = aapi.zeros((16, 16), dtype=aapi.int32, **kwargs)
            b = aapi.ones((12, 12), dtype=aapi.int32, **kwargs)
            a[3:13, 3:13] = b[0:10, 1:11] + b[1:11, 1:11] * b[1, 1]
            return a

        assert runAndCompare(doit)

    def test_add_shifted1(self):
        for dtyp in mpi_idtypes:
            aa = sp.ones((16, 16), dtype=dtyp, device=device)
            bb = sp.ones((16, 16), dtype=dtyp, device=device)
            a = aa[0:8, 0:16]
            b = bb[5:13, 0:16]
            c = a + b + 1
            r1 = sp.sum(c, [0, 1])
            v = 8 * 16 * 3
            assert float(r1) == v

    @pytest.mark.skip(reason="FIXME reshape")
    def test_add_shifted2(self):
        def doit(aapi, **kwargs):
            a = aapi.reshape(
                aapi.arange(0, 64, 1, dtype=aapi.float32, **kwargs), [8, 8]
            )
            b = aapi.reshape(
                aapi.arange(0, 64, 1, dtype=aapi.float32, **kwargs), [8, 8]
            )
            c = a[2:6, 0:8]
            d = b[0:8:2, 0:8]
            return c + d

        assert runAndCompare(doit)

    def test_add_shifted3(self):
        def doit():
            aa = sp.arange(0, 64, 1, dtype=sp.int32, device=device)
            bb = sp.arange(0, 64, 1, dtype=sp.int32, device=device)
            a = aa[0:8]
            b = bb[50:58]
            c = a + b + 1
            return sp.sum(c)

        r1 = doit()
        assert int(r1) == 464

    def test_add_shifted4(self):
        aa = sp.arange(0, 64, 1, dtype=sp.int32, device=device)
        a = aa[0:8]
        b = aa[50:58]
        c = a + b + 1
        r1 = sp.sum(c)
        assert int(r1) == 464

    def test_add_shifted5(self):
        aa = sp.arange(0, 64, 1, dtype=sp.int32, device=device)
        bb = sp.arange(0, 64, 1, dtype=sp.int32, device=device)
        a = aa[0:16:2]
        b = bb[40:64:3]
        c = a + b + 1
        r1 = sp.sum(c)
        assert int(r1) == 468

    def test_add_shifted6(self):
        aa = sp.arange(0, 64, 1, dtype=sp.int32, device=device)
        a = aa[0:16:2]
        b = aa[30:54:3]
        c = a + b + 1
        r1 = sp.sum(c)
        assert int(r1) == 388

    def test_add_broadcast(self):
        def doit(aapi, **kwargs):
            a = aapi.zeros((16, 16), dtype=aapi.int32, **kwargs)
            b = aapi.arange(1, 16, 1, dtype=aapi.int32, **kwargs)
            a[3:13, 3:13] = a[0:10, 1:11] + b[0]
            return a

        assert runAndCompare(doit)

    @pytest.mark.skip(reason="FIXME")
    def test_prod_het(self):
        a = sp.full([16, 16], 2, sp.float32, device=device)
        b = sp.full([16, 16], 2, sp.int32, device=device)
        c = a * b
        r = sp.sum(c, [0, 1])
        v = 16 * 16 * 2 * 2
        assert float(r) == v

    @pytest.mark.skipif(len(device), reason="FIXME power on GPU")
    def test_pow(self):
        for dtyp in [sp.int32, sp.float32]:
            a = sp.full((6, 6), 3, dtype=dtyp, device=device)
            b = sp.full((6, 6), 2, dtype=dtyp, device=device)
            c = a**b
            r1 = sp.sum(c, [0, 1])
            v = 6 * 6 * 9
            assert float(r1) == v

    def test_bitwise_and(self):
        for dtyp in mpi_idtypes:
            a = sp.full((6, 6), 3, dtype=dtyp, device=device)
            b = sp.full((6, 6), 2, dtype=dtyp, device=device)
            c = a & b
            r1 = sp.sum(c, [0, 1])
            v = 6 * 6 * 2
            assert float(r1) == v

    @pytest.mark.skipif(len(device), reason="FIXME 64bit on GPU")
    def test_add_typecast(self):
        type_list = [
            (sp.int8, sp.int64, sp.int64),
            (sp.int64, sp.int8, sp.int64),
            (sp.int8, sp.float64, sp.float64),
            (sp.float64, sp.int8, sp.float64),
            (sp.float32, sp.float64, sp.float64),
            (sp.float32, sp.int64, sp.float64),
            (sp.int32, sp.uint32, sp.int64),
            (sp.int32, sp.uint64, sp.int64),
        ]
        for atype, btype, ctype in type_list:
            a = sp.arange(0, 8, 1, dtype=atype, device=device)
            b = sp.ones((8,), dtype=btype, device=device)
            c = a + b
            assert c.dtype == ctype
            c2 = sp.to_numpy(c)
            assert numpy.allclose(c2, [1, 2, 3, 4, 5, 6, 7, 8])

    def test_reflected_sub(self):
        a = sp.full((4,), 4, dtype=sp.float32, device=device)
        b = 10.0
        c = b - a
        c2 = sp.to_numpy(c)
        assert numpy.allclose(c2, [6, 6, 6, 6])

    def test_reflected_div(self):
        a = sp.full((4,), 2, dtype=sp.float32, device=device)
        b = 10.0
        c = b / a
        c2 = sp.to_numpy(c)
        assert numpy.allclose(c2, [5, 5, 5, 5])
