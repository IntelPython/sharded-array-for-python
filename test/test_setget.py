from utils import runAndCompare
import pytest


class TestSetGet:
    @pytest.mark.skip(reason="FIXME")
    def test_get_item(self):
        for i in range(1, 5):
            N = (10**i) * 16
            print(N, N // 4, (N // 4) * 3)
            doit = lambda aapi: aapi.ones((N, N), aapi.float64)[
                0:N:2, N // 4 : (N // 4) * 3 : 2
            ]

            assert runAndCompare(doit)

    def test_setitem1(self):
        def doit(aapi):
            a = aapi.ones((16, 16), aapi.float64)
            a[0:16:2, 0:16:2] = aapi.zeros([8, 8], aapi.float64)
            return a

        assert runAndCompare(doit)

    def test_setitem2(self):
        def doit(aapi):
            a = aapi.ones((16, 16), aapi.float64)
            b = aapi.zeros((16, 16), aapi.float64)
            a[1:8, 0:6] = b[0:7, 0:6]
            return a

        assert runAndCompare(doit)

    def test_setitem3(self):
        def doit(aapi):
            a = aapi.ones((16, 16), aapi.float64)
            b = aapi.zeros((16, 16), aapi.float64)
            a[7:16:3, 4:10:2] = b[4:7, 10:16:2]
            return a

        assert runAndCompare(doit)
