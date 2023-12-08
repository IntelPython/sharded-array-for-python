import sharpy as sp
from utils import runAndCompare, mpi_dtypes
import pytest


class TestEWU:
    def test_sqrt(self):
        a = sp.full((16, 16), 9, sp.float32)
        c = sp.sum(sp.sqrt(a), [0, 1])
        v = 16 * 16 * 3
        assert float(c) == v

    @pytest.mark.skip(reason="FIXME")
    def test_equal(self):
        a = sp.full((16, 16), 9, sp.float32)
        b = sp.full((16, 16), 9, sp.float32)
        c = a == b
        assert c.dtype == sp.bool
