import ddptensor as dt
import numpy as np

class TestEWB:
    def test_add1(self):
        a = dt.ones([16,16], dtype=dt.float64)
        b = dt.ones([16,16], dtype=dt.float64)
        c = a + b
        r1 = dt.sum(c, [0,1])
        v = 16*16*2
        assert float(r1) == v

    def test_add2(self):
        a = dt.ones([16,16], dtype=dt.float64)
        c = a + 1
        r1 = dt.sum(c, [0,1])
        v = 16*16*2
        assert float(r1) == v

    def test_add3(self):
        a = dt.ones([16,16], dtype=dt.float64)
        b = dt.ones([16,16], dtype=dt.float64)
        c = a + b + 1
        r1 = dt.sum(c, [0,1])
        v = 16*16*3
        assert float(r1) == v

    def test_add_shifted1(self):
        aa = dt.ones([16,16], dtype=dt.float64)
        bb = dt.ones([16,16], dtype=dt.float64)
        a = aa[0:8, 0:16]
        b = bb[5:13, 0:16]
        c = a + b + 1
        r1 = dt.sum(c, [0,1])
        v = 8*16*3
        assert float(r1) == v

    def test_add_shifted2(self):
        def _do(impl):
            a = impl.reshape(impl.arange(0,64,1, dtype=impl.float64), [8,8])
            b = impl.reshape(impl.arange(0,64,1, dtype=impl.float64), [8,8])
            c = a[2:6, 0:8]
            d = b[0:8:2, 0:8]
            return c + d
        assert float(dt.sum(_do(dt), [0,1])) == float(np.sum(_do(np)))

    def test_prod_het(self):
        a = dt.full([16,16], 2, dt.float64)
        b = dt.full([16,16], 2, dt.int64)
        c = a * b
        r = dt.sum(c, [0,1])
        v = 16*16*2*2
        assert float(r) == v
