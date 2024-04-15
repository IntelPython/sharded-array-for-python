import numpy as np
import pytest
from mpi4py import MPI
from utils import device

import sharpy as sp
from sharpy import _sharpy_cw


@pytest.mark.skipif(_sharpy_cw, reason="Only applicable to SPMD mode")
class TestSPMD:
    @pytest.mark.skip(reason="FIXME")
    def test_get_slice(self):
        a = sp.ones(
            (2 + MPI.COMM_WORLD.size * 2, 2 + MPI.COMM_WORLD.size * 2),
            sp.float32,
            device=device,
        )
        MPI.COMM_WORLD.barrier()
        b = sp.spmd.get_slice(
            a,
            (
                slice(MPI.COMM_WORLD.rank, MPI.COMM_WORLD.rank * 2),
                slice(1, 2 + MPI.COMM_WORLD.rank),
            ),
        )
        c = np.sum(b)
        v = MPI.COMM_WORLD.rank * (1 + MPI.COMM_WORLD.rank)
        assert c == v
        MPI.COMM_WORLD.barrier()

    def test_get_locals(self):
        a = sp.ones((32, 32), sp.float32, device=device)
        la = sp.spmd.get_locals(a)[0]
        la[0, 0] = 0
        MPI.COMM_WORLD.barrier()
        c = sp.sum(a, [0, 1])
        v = 32 * 32 - MPI.COMM_WORLD.size
        assert float(c) == v
        MPI.COMM_WORLD.barrier()

    @pytest.mark.skip(reason="FIXME imex-remove-temporaries")
    def test_get_locals_of_view(self):
        a = sp.ones((32, 32), sp.float32, device=device)
        b = a[0:32:2, 0:32:2]
        lb = sp.spmd.get_locals(b)[0]
        assert len(lb) > 0
        lb[0, 0] = 0
        MPI.COMM_WORLD.barrier()
        c = sp.sum(a, [0, 1])
        v = 32 * 32 - MPI.COMM_WORLD.size
        assert float(c) == v
        MPI.COMM_WORLD.barrier()

    def test_gather1(self):
        a = sp.reshape(
            sp.arange(0, 110, 1, dtype=sp.float32, device=device), [11, 10]
        )
        b = sp.spmd.gather(a)
        c = np.sum(b)
        v = np.sum(np.reshape(np.arange(0, 110, 1, dtype=np.float32), (11, 10)))
        assert float(c) == v
        MPI.COMM_WORLD.barrier()

    def test_gather2(self):
        a = sp.arange(0, 110, 1, dtype=sp.float32, device=device)
        b = sp.spmd.gather(a)
        c = np.sum(b)
        v = np.sum(np.arange(0, 110, 1, dtype=np.float32))
        assert float(c) == v
        MPI.COMM_WORLD.barrier()

    def test_gather_0d(self):
        a = sp.full((), 5, dtype=sp.int32, device=device)
        b = sp.spmd.gather(a)
        assert float(b) == 5
        MPI.COMM_WORLD.barrier()

    def test_gather_strided1(self):
        a = sp.reshape(
            sp.arange(0, 110, 1, dtype=sp.float32, device=device), [11, 10]
        )
        b = sp.spmd.gather(a[4:12:2, 1:11:3])
        c = np.sum(b)
        v = np.sum(
            np.reshape(np.arange(0, 110, 1, dtype=np.float32), (11, 10))[
                4:12:2, 1:11:3
            ]
        )
        assert float(c) == v
        MPI.COMM_WORLD.barrier()

    def test_gather_strided2(self):
        a = sp.arange(0, 110, 1, dtype=sp.float32, device=device)
        b = sp.spmd.gather(a[34:82:2])
        c = np.sum(b)
        v = np.sum(np.arange(0, 110, 1, dtype=np.float32)[34:82:2])
        assert float(c) == v
        MPI.COMM_WORLD.barrier()

    @pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="FIXME multi-proc")
    def test_from_locals(self):
        npa = np.arange(1, 11, 2, dtype=np.int64)
        a = sp.spmd.from_locals(npa)
        x = 4711
        npa[0] = x
        assert int(a[0]) == x
