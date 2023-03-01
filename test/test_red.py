import numpy as np
import ddptensor as dt
from misc import arange_reshape


class TestRed:
    def test_sum(self):
        a = arange_reshape(0, 64, 1, (8, 8))
        r1 = dt.sum(dt.sum(a, [1]), [0])
        r2 = dt.sum(a, [0, 1])
        v = np.sum(np.arange(64))
        assert float(r1) == v
        assert float(r2) == v
