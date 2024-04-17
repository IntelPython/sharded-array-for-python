from utils import runAndCompare


class TestRed:
    def test_sum(self):
        def doit(aapi, **kwargs):
            a = aapi.arange(0, 64, 1, dtype=aapi.int64, **kwargs)
            b = aapi.reshape(a, (8, 8))
            return aapi.sum(b)

        assert runAndCompare(doit, False)

    def test_max(self):
        def doit(aapi, **kwargs):
            a = aapi.linspace(
                0.1, 64.3, 111, dtype=aapi.float32, endpoint=True, **kwargs
            )
            return aapi.max(a[6:66])

        assert runAndCompare(doit, False)
