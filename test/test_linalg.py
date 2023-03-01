import numpy as np
from mpi4py import MPI
import ddptensor as dt


class TestLinAlg:
    def test_vecdot1(self):
        def gen(m):
            a = m.arange(1, 36, 1, dtype=m.float64)
            b = m.arange(1, 22, 1, dtype=m.float64)
            return m.reshape(a, (5, 7)), m.reshape(b, (7, 3))

        a, b = gen(dt)
        c = float(dt.sum(dt.vecdot(a, b, 0), [0, 1]))
        a, b = gen(np)
        v = float(np.sum(np.dot(a, b)))
        assert c == v
