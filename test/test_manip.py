from utils import runAndCompare


class TestManip:
    def test_reshape1(self):
        def doit(aapi):
            a = aapi.arange(0, 12 * 11, 1, aapi.int64)
            return aapi.reshape(a, [6, 22])

        assert runAndCompare(doit)

    def test_reshape2(self):
        def doit(aapi):
            a = aapi.arange(0, 12 * 11, 1, aapi.int64)
            b = aapi.reshape(a, [12, 11])
            c = b[0:12:2, 0:10:2]
            return aapi.reshape(c, [5, 6])

        assert runAndCompare(doit)
