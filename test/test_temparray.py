from utils import device

import sharpy as sp


class TestTemporaries:
    def test_temp_inline(self):
        dtyp = sp.float32
        a = sp.ones((6, 6), dtype=dtyp, device=device)
        b = sp.ones((6, 6), dtype=dtyp, device=device)

        sp.sync()
        c = a[0:5, 0:5]
        b[0:5, 0:5] = b[0:5, 0:5] + c
        del c
        sp.sync()

        c = a[0:5, 0:5]
        b[0:5, 0:5] = b[0:5, 0:5] + c
        del c
        sp.sync()

        r1 = sp.sum(b, [0, 1])
        v = 5 * 5 * 3 + 6 + 5
        assert float(r1) == v

    def test_temp_func(self):
        def func(a):
            return a[0:5, 0:5]

        def update(a, b):
            c = func(a)
            b[0:5, 0:5] = b[0:5, 0:5] + c

        dtyp = sp.float32
        a = sp.ones((6, 6), dtype=dtyp, device=device)
        b = sp.ones((6, 6), dtype=dtyp, device=device)

        sp.sync()
        update(a, b)
        sp.sync()
        update(a, b)
        sp.sync()

        r1 = sp.sum(b, [0, 1])
        v = 5 * 5 * 3 + 6 + 5
        assert float(r1) == v
