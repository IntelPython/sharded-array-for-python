import sharpy as sp
import pytest


class TestIEWB:
    @pytest.mark.skip(reason="FIXME")
    def test_add1(self):
        a = sp.ones([16, 16], dtype=sp.float64)
        b = sp.ones([16, 16], dtype=sp.float64)
        a += b
        r1 = sp.sum(a, [0, 1])
        v = 16 * 16 * 2
        assert float(r1) == v

    @pytest.mark.skip(reason="FIXME")
    def test_add2(self):
        a = sp.ones([16, 16], dtype=sp.float64)
        a += 1
        r1 = sp.sum(a, [0, 1])
        v = 16 * 16 * 2
        assert float(r1) == v

    @pytest.mark.skip(reason="FIXME")
    def test_add3(self):
        a = sp.ones([16, 16], dtype=sp.float64)
        b = sp.ones([16, 16], dtype=sp.float64)
        a += b + 1
        r1 = sp.sum(a, [0, 1])
        v = 16 * 16 * 3
        assert float(r1) == v
