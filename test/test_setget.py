from utils import runAndCompare


class TestSetGet:
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
