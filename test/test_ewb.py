import ddptensor as dt
from utils import runAndCompare, mpi_dtypes
import pytest

mpi_idtypes = [  # FIXME
    dt.int64,
    dt.uint64,
    # dt.int32,
    # dt.uint32,
    # dt.int8,
    # dt.uint8,
]


class TestEWB:
    def test_add1(self):
        for dtyp in mpi_dtypes:
            a = dt.ones((6, 6), dtype=dtyp)
            b = dt.ones((6, 6), dtype=dtyp)
            c = a + b
            r1 = dt.sum(c)
            v = 6 * 6 * 2
            assert float(r1) == v

    def test_add2(self):
        for dtyp in mpi_idtypes:
            a = dt.ones((16, 16), dtype=dtyp)
            c = a + 1
            r1 = dt.sum(c, [0, 1])
            v = 16 * 16 * 2
            assert float(r1) == v

    def test_add3(self):
        for dtyp in mpi_idtypes:
            a = dt.ones((16, 16), dtype=dtyp)
            b = dt.ones((16, 16), dtype=dtyp)
            c = a + b + 1
            r1 = dt.sum(c, [0, 1])
            v = 16 * 16 * 3
            assert float(r1) == v

    def test_add_shifted1(self):
        for dtyp in mpi_idtypes:
            aa = dt.ones((16, 16), dtype=dtyp)
            bb = dt.ones((16, 16), dtype=dtyp)
            a = aa[0:8, 0:16]
            b = bb[5:13, 0:16]
            c = a + b + 1
            r1 = dt.sum(c, [0, 1])
            v = 8 * 16 * 3
            assert float(r1) == v

    def test_add_shifted2(self):
        def doit(aapi):
            a = aapi.reshape(aapi.arange(0, 64, 1, dtype=aapi.float64), [8, 8])
            b = aapi.reshape(aapi.arange(0, 64, 1, dtype=aapi.float64), [8, 8])
            c = a[2:6, 0:8]
            d = b[0:8:2, 0:8]
            return c + d

        assert runAndCompare(doit)

    @pytest.mark.skip(reason="FIXME")
    def test_prod_het(self):
        a = dt.full([16, 16], 2, dt.float64)
        b = dt.full([16, 16], 2, dt.int64)
        c = a * b
        r = dt.sum(c, [0, 1])
        v = 16 * 16 * 2 * 2
        assert float(r) == v

    def test_pow(self):
        for dtyp in [dt.int64, dt.float64]:
            a = dt.full((6, 6), 3, dtype=dtyp)
            b = dt.full((6, 6), 2, dtype=dtyp)
            c = a**b
            r1 = dt.sum(c, [0, 1])
            v = 6 * 6 * 9
            assert float(r1) == v

    def test_bitwise_and(self):
        for dtyp in mpi_idtypes:
            a = dt.full((6, 6), 3, dtype=dtyp)
            b = dt.full((6, 6), 2, dtype=dtyp)
            c = a & b
            r1 = dt.sum(c, [0, 1])
            v = 6 * 6 * 2
            assert float(r1) == v
