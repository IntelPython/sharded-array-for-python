import ddptensor as dt

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

    def test_prod_het(self):
        a = dt.full([16,16], 2, dt.float64)
        b = dt.full([16,16], 2, dt.int64)
        c = a * b
        r = dt.sum(c, [0,1])
        v = 16*16*2*2
        assert float(r) == v
