import numpy as np
import ddptensor as dt
import pytest


class TestIO:
    @pytest.mark.skip(reason="FIXME reshape")
    def test_to_numpy2d(self):
        a = dt.reshape(dt.arange(0, 110, 1, dtype=dt.float32), [11, 10])
        b = dt.to_numpy(a)
        c = np.sum(b)
        v = np.sum(np.reshape(np.arange(0, 110, 1, dtype=np.float32), (11, 10)))
        assert float(c) == v

    def test_to_numpy1d(self):
        a = dt.arange(0, 110, 1, dtype=dt.float32)
        b = dt.to_numpy(a)
        c = np.sum(b)
        v = np.sum(np.arange(0, 110, 1, dtype=np.float32))
        assert float(c) == v

    @pytest.mark.skip(reason="FIXME reshape")
    def test_to_numpy_strided(self):
        a = dt.reshape(dt.arange(0, 110, 1, dtype=dt.float32), [11, 10])
        b = dt.to_numpy(a[4:12:2, 1:11:3])
        c = np.sum(b)
        v = np.sum(
            np.reshape(np.arange(0, 110, 1, dtype=np.float32), (11, 10))[4:12:2, 1:11:3]
        )
        assert float(c) == v
