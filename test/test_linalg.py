import numpy as np
import sharpy as sp
import pytest


class TestLinAlg:
    @pytest.mark.skip(reason="FIXME")
    def test_vecdot1(self):
        def gen(m):
            a = m.arange(1, 36, 1, dtype=m.float64)
            b = m.arange(1, 22, 1, dtype=m.float64)
            return m.reshape(a, (5, 7)), m.reshape(b, (7, 3))

        a, b = gen(dt)
        c = float(sp.sum(sp.vecdot(a, b, 0), [0, 1]))
        a, b = gen(np)
        v = float(np.sum(np.dot(a, b)))
        assert c == v
