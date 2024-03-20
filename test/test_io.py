import numpy as np
import pytest
from utils import device

import sharpy as sp


class TestIO:
    @pytest.mark.parametrize("backend", ["opencl:", "level-zero:", "cuda:", ""])
    @pytest.mark.parametrize("device", ["host", "gpu", "cpu", "accelerator"])
    @pytest.mark.parametrize("subdev", [":2", ""])
    def test_device_input_valid(self, backend, device, subdev):
        a = sp.ones((4,), device=f"{backend}{device}{subdev}")
        assert a.size == 4

    @pytest.mark.parametrize(
        "device",
        [
            "opencl3:gpu",
            "opencl:hostx",
            "opencl:cpu:r",
            "opencl:gpu::1",
            "opencl::gpu:1",
        ],
    )
    def test_device_input_invalid(self, device):
        with pytest.raises(ValueError, match="Invalid device string: *"):
            sp.ones((4,), device=device)

    @pytest.mark.skip(reason="FIXME reshape")
    def test_to_numpy2d(self):
        a = sp.reshape(
            sp.arange(0, 110, 1, dtype=sp.float32, device=device), [11, 10]
        )
        b = sp.to_numpy(a)
        c = np.sum(b)
        v = np.sum(np.reshape(np.arange(0, 110, 1, dtype=np.float32), (11, 10)))
        assert float(c) == v

    def test_to_numpy1d(self):
        a = sp.arange(0, 110, 1, dtype=sp.float32, device=device)
        b = sp.to_numpy(a)
        c = np.sum(b)
        v = np.sum(np.arange(0, 110, 1, dtype=np.float32))
        assert float(c) == v

    @pytest.mark.skip(reason="FIXME reshape")
    def test_to_numpy_strided(self):
        a = sp.reshape(
            sp.arange(0, 110, 1, dtype=sp.float32, device=device), [11, 10]
        )
        b = sp.to_numpy(a[4:12:2, 1:11:3])
        c = np.sum(b)
        v = np.sum(
            np.reshape(np.arange(0, 110, 1, dtype=np.float32), (11, 10))[
                4:12:2, 1:11:3
            ]
        )
        assert float(c) == v
