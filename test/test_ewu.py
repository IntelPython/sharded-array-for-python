import sharpy as sp
import pytest
import numpy

type_list = [sp.int64, sp.float64]


class TestEWU:
    def test_sqrt(self):
        a = sp.full((16, 16), 9, sp.float32)
        c = sp.sum(sp.sqrt(a), [0, 1])
        v = 16 * 16 * 3
        assert float(c) == v

    def test_abs(self):
        a = sp.full((6,), -5, sp.float32)
        b = sp.abs(a)
        assert numpy.allclose(sp.to_numpy(b), [5, 5, 5, 5, 5, 5])

    def test_negative(self):
        for dtype in type_list:
            a = sp.arange(0, 6, 1, dtype)
            b = -a
            assert numpy.allclose(sp.to_numpy(b), [0, -1, -2, -3, -4, -5])

    def test_positive(self):
        for dtype in type_list:
            a = sp.arange(0, 6, 1, dtype)
            b = +a
            assert numpy.allclose(sp.to_numpy(b), [0, 1, 2, 3, 4, 5])

    @pytest.mark.skip(reason="FIXME")
    def test_equal(self):
        a = sp.full((16, 16), 9, sp.float32)
        b = sp.full((16, 16), 9, sp.float32)
        c = a == b
        assert c.dtype == sp.bool
