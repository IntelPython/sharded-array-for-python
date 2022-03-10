import ddptensor as dt

class TestEWB:
    def test_sqrt(self):
        a = dt.full([16,16], 9, dt.float64)
        c = dt.sum(dt.sqrt(a), [0,1])
        v = 16*16*3
        assert float(c) == v

    def test_equal(self):
        a = dt.full([16,16], 9, dt.float64)
        b = dt.full([16,16], 9, dt.float64)
        c = (a == b)
        assert c.dtype == dt.bool
