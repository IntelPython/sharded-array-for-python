import numpy as np
from mpi4py import MPI
import ddptensor as dt
from ddptensor import _ddpt_cw
import pytest


@pytest.mark.skipif(_ddpt_cw, reason="Only applicable to SPMD mode")
class TestSPMD:
    @pytest.mark.skip(reason="FIXME")
    def test_get_slice(self):
        a = dt.ones(
            (2 + MPI.COMM_WORLD.size * 2, 2 + MPI.COMM_WORLD.size * 2), dt.float64
        )
        MPI.COMM_WORLD.barrier()
        b = dt.spmd.get_slice(
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

    def test_get_local(self):
        a = dt.ones((32, 32), dt.float64)
        l = dt.spmd.get_local(a)
        l[0, 0] = 0
        MPI.COMM_WORLD.barrier()
        c = dt.sum(a, [0, 1])
        v = 32 * 32 - MPI.COMM_WORLD.size
        assert float(c) == v
        MPI.COMM_WORLD.barrier()

    @pytest.mark.skip(reason="FIXME")
    def test_get_local_of_view(self):
        a = dt.ones((32, 32), dt.float64)
        b = a[0:32:2, 0:32:2]
        l = dt.spmd.get_local(b)
        assert len(l) > 0
        l[0, 0] = 0
        MPI.COMM_WORLD.barrier()
        c = dt.sum(a, [0, 1])
        v = 32 * 32 - MPI.COMM_WORLD.size
        assert float(c) == v
        MPI.COMM_WORLD.barrier()

    def test_gather(self):
        a = dt.reshape(dt.arange(0, 110, 1, dtype=dt.float64), [11, 10])
        b = dt.spmd.gather(a)
        c = np.sum(b)
        v = np.sum(np.reshape(np.arange(0, 110, 1, dtype=np.float64), (11, 10)))
        assert float(c) == v
        MPI.COMM_WORLD.barrier()

    @pytest.mark.skip(reason="FIXME")
    def test_gather_strided(self):
        a = dt.reshape(dt.arange(0, 110, 1, dtype=dt.float64), [11, 10])
        b = dt.spmd.gather(a[4:12:2, 1:11:3])
        c = np.sum(b)
        v = np.sum(
            np.reshape(np.arange(0, 110, 1, dtype=np.float64), (11, 10))[4:12:2, 1:11:3]
        )
        assert float(c) == v
        MPI.COMM_WORLD.barrier()
