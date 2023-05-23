import ddptensor as dt
import pytest


class TestTemporaries:
    def test_temp_inline(self):
        dtyp = dt.float64
        a = dt.ones((6, 6), dtype=dtyp)
        b = dt.ones((6, 6), dtype=dtyp)

        dt.sync()
        c = a[0:5, 0:5]
        b[0:5, 0:5] = b[0:5, 0:5] + c
        del c
        dt.sync()

        c = a[0:5, 0:5]
        b[0:5, 0:5] = b[0:5, 0:5] + c
        del c
        dt.sync()

        r1 = dt.sum(b, [0, 1])
        v = 5 * 5 * 3 + 6 + 5
        assert float(r1) == v

    def test_temp_func(self):
        def func(a):
            return a[0:5, 0:5]

        def update(a, b):
            c = func(a)
            b[0:5, 0:5] = b[0:5, 0:5] + c

        dtyp = dt.float64
        a = dt.ones((6, 6), dtype=dtyp)
        b = dt.ones((6, 6), dtype=dtyp)

        dt.sync()
        update(a, b)
        dt.sync()
        update(a, b)
        dt.sync()

        r1 = dt.sum(b, [0, 1])
        v = 5 * 5 * 3 + 6 + 5
        assert float(r1) == v
