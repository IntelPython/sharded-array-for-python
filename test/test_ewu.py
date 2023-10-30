import ddptensor as dt
from utils import runAndCompare, mpi_dtypes
import pytest


class TestEWU:
    def test_sqrt(self):
        a = dt.full((16, 16), 9, dt.float32)
        c = dt.sum(dt.sqrt(a), [0, 1])
        v = 16 * 16 * 3
        assert float(c) == v

    @pytest.mark.skip(reason="FIXME")
    def test_equal(self):
        a = dt.full((16, 16), 9, dt.float32)
        b = dt.full((16, 16), 9, dt.float32)
        c = a == b
        assert c.dtype == dt.bool
