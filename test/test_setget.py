import sharpy as sp
import numpy
from utils import runAndCompare
import pytest


class TestSetGet:
    @pytest.mark.skip(reason="FIXME")
    def test_get_item(self):
        for i in range(1, 5):
            N = (10**i) * 16
            print(N, N // 4, (N // 4) * 3)
            doit = lambda aapi: aapi.ones((N, N), aapi.float32)[
                0:N:2, N // 4 : (N // 4) * 3 : 2
            ]

            assert runAndCompare(doit)

    def test_setitem1(self):
        def doit(aapi):
            a = aapi.ones((16, 16), aapi.float32)
            a[0:16:2, 0:16:2] = aapi.zeros([8, 8], aapi.float32)
            return a

        assert runAndCompare(doit)

    def test_setitem2(self):
        def doit(aapi):
            a = aapi.ones((16, 16), aapi.float32)
            b = aapi.fromfunction(lambda i, j: 10 * i + j, (16, 16), dtype=aapi.float32)
            a[1:8, 0:6] = b[0:7, 0:6]
            return a

        assert runAndCompare(doit)

    def test_setitem3(self):
        def doit(aapi):
            a = aapi.ones((16, 16), aapi.float32)
            b = aapi.fromfunction(lambda i, j: 10 * i + j, (16, 16), dtype=aapi.float32)
            a[7:16:3, 4:10:2] = b[4:7, 10:16:2]
            return a

        assert runAndCompare(doit)

    def test_setitem4(self):
        # Note: test halo update without send buffer
        def doit(aapi):
            a = aapi.ones((16, 16), aapi.float32)
            b = aapi.fromfunction(lambda i, j: 10 * i + j, (16, 16), dtype=aapi.float32)
            a[7:16:3, 0:16] = b[4:7, 0:16]
            return a

        assert runAndCompare(doit)

    def test_setitem5(self):
        # Note: test assignment to one full local part
        def doit(aapi):
            a = aapi.fromfunction(lambda i, j: 10 * i + j, (16, 16), dtype=aapi.int32)
            a[0:10, 4:11] = a[0:10, 4:11]
            return a

        assert runAndCompare(doit)

    def test_setitem6(self):
        def doit(aapi):
            n = 16
            a = aapi.fromfunction(lambda i, j: i, (n, n), dtype=aapi.float32)
            b = aapi.zeros((n + 1, n + 1), aapi.float32)

            b[1:n, 1:n] = a[1:n, 1:n]
            return b

        assert runAndCompare(doit)

    def test_setitem7(self):
        # Note: assert halo does not segfault
        def doit(aapi):
            n = 1024
            a = aapi.fromfunction(lambda i, j: i, (n, n), dtype=aapi.float32)
            b = aapi.zeros((n, n), aapi.float32)

            b[1:n, 1:n] = a[1:n, 1:n]
            b[0, 1:n] = a[0, 1:n]

            return b

        assert runAndCompare(doit)

    def test_setitem8(self):
        # Note: assert halo does not segfault
        def doit(aapi):
            n = 1024
            a = aapi.fromfunction(lambda i, j: i, (n, n), dtype=aapi.float64)
            b = aapi.zeros((n, n), aapi.float64)

            b[0, 1:n] = 1.0 * a[0, 1:n]
            b[1 : n - 1, 1:n] = 1.0 * a[1 : n - 1, 1:n]
            b[n - 1, 1:n] = 1.0 * a[n - 1, 1:n]
            return b

        assert runAndCompare(doit)

    def test_colon(self):
        a = sp.ones((16, 16), sp.float32)
        b = sp.zeros((16, 16), sp.float32)
        a[:, :] = b[:, :]
        r1 = sp.sum(a)
        assert float(r1) == 0

    def test_neg_last(self):
        a = sp.arange(0, 16, 1, sp.int64)
        b = a[-1]
        assert int(b) == 15

    def test_neg_end(self):
        def doit(aapi):
            a = aapi.arange(0, 16, 1, aapi.int64)
            b = aapi.arange(2, 18, 1, aapi.int64)
            a[0:10] = b[:-6]
            return a

        assert runAndCompare(doit)

    def test_neg_start(self):
        def doit(aapi):
            a = aapi.arange(0, 16, 1, aapi.int64)
            b = aapi.arange(2, 18, 1, aapi.int64)
            a[0:4] = b[-4:]
            return a

        assert runAndCompare(doit)

    def test_neg_slice(self):
        def doit(aapi):
            a = aapi.ones((16, 16), aapi.float64)
            b = aapi.zeros((16, 16), aapi.float64)
            a[:-1, 0:3] = b[:-1, -4:-1]
            return a

        assert runAndCompare(doit)

    @pytest.mark.skip(reason="FIXME multi-proc")
    def test_neg_stride(self):
        def doit(aapi):
            a = aapi.arange(0, 8, 1, aapi.int64)
            b = a[8:0:-1]
            return b

        assert runAndCompare(doit)

    @pytest.mark.skip(reason="FIXME")
    def test_reverse(self):
        def doit(aapi):
            a = aapi.arange(0, 8, 1, aapi.int64)
            b = a[::-1]
            return b

        assert runAndCompare(doit)

    def test_assign_bcast_scalar(self):
        a = sp.zeros((16, 16), sp.int64)
        b = 2
        a[:, :] = b
        a2 = sp.to_numpy(a)
        assert numpy.all(a2 == 2)
