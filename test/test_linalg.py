import numpy as np
import pytest
from utils import device

import sharpy as sp


class TestLinAlg:
    @pytest.mark.skip(reason="FIXME")
    def test_vecdot1(self):
        def gen(m):
            a = m.arange(1, 36, 1, dtype=m.float64, device=device)
            b = m.arange(1, 22, 1, dtype=m.float64, device=device)
            return m.reshape(a, (5, 7)), m.reshape(b, (7, 3))

        a, b = gen(sp)
        c = float(sp.sum(sp.vecdot(a, b, 0), [0, 1]))
        a, b = gen(np)
        v = float(np.sum(np.dot(a, b)))
        assert c == v
