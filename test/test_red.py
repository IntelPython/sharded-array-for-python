import sharpy as sp
from utils import runAndCompare
import pytest


class TestRed:
    @pytest.mark.skip(reason="FIXME reshape")
    def test_sum(self):
        def doit(aapi):
            a = aapi.arange(0, 64, 1, dtype=aapi.int64)
            b = aapi.reshape(a, (8, 8))
            return aapi.sum(a)

        assert runAndCompare(doit, False)

    def test_max(self):
        def doit(aapi):
            a = aapi.linspace(0.1, 64.3, 111, dtype=aapi.float32, endpoint=True)
            return aapi.max(a[6:66])

        assert runAndCompare(doit, False)
