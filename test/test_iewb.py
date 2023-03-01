import ddptensor as dt


class TestIEWB:
    def test_add1(self):
        a = dt.ones([16, 16], dtype=dt.float64)
        b = dt.ones([16, 16], dtype=dt.float64)
        a += b
        r1 = dt.sum(a, [0, 1])
        v = 16 * 16 * 2
        assert float(r1) == v

    def test_add2(self):
        a = dt.ones([16, 16], dtype=dt.float64)
        a += 1
        r1 = dt.sum(a, [0, 1])
        v = 16 * 16 * 2
        assert float(r1) == v

    def test_add3(self):
        a = dt.ones([16, 16], dtype=dt.float64)
        b = dt.ones([16, 16], dtype=dt.float64)
        a += b + 1
        r1 = dt.sum(a, [0, 1])
        v = 16 * 16 * 3
        assert float(r1) == v
