from utils import device

import sharpy as sp


class TestIEWB:
    def test_add1(self):
        a = sp.ones([16, 16], dtype=sp.float32, device=device)
        b = sp.ones([16, 16], dtype=sp.float32, device=device)
        a += b
        r1 = sp.sum(a, [0, 1])
        v = 16 * 16 * 2
        assert float(r1) == v

    def test_add2(self):
        a = sp.ones([16, 16], dtype=sp.float32, device=device)
        a += 1
        r1 = sp.sum(a, [0, 1])
        v = 16 * 16 * 2
        assert float(r1) == v

    def test_add3(self):
        a = sp.ones([16, 16], dtype=sp.float32, device=device)
        b = sp.ones([16, 16], dtype=sp.float32, device=device)
        a += b + 1
        r1 = sp.sum(a, [0, 1])
        v = 16 * 16 * 3
        assert float(r1) == v
