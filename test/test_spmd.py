import numpy as np
from mpi4py import MPI
import ddptensor as dt
from ddptensor import _ddpt_cw
import pytest
import os


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

    def test_get_locals(self):
        a = dt.ones((32, 32), dt.float64)
        l = dt.spmd.get_locals(a)[0]
        l[0, 0] = 0
        MPI.COMM_WORLD.barrier()
        c = dt.sum(a, [0, 1])
        v = 32 * 32 - MPI.COMM_WORLD.size
        assert float(c) == v
        MPI.COMM_WORLD.barrier()

    @pytest.mark.skipif(
        MPI.COMM_WORLD.size == 1 and os.getenv("DDPT_FORCE_DIST", "") == "",
        reason="FIXME extra memref.copy",
    )
    def test_get_locals_of_view(self):
        a = dt.ones((32, 32), dt.float64)
        b = a[0:32:2, 0:32:2]
        l = dt.spmd.get_locals(b)[0]
        assert len(l) > 0
        l[0, 0] = 0
        MPI.COMM_WORLD.barrier()
        c = dt.sum(a, [0, 1])
        v = 32 * 32 - MPI.COMM_WORLD.size
        assert float(c) == v
        MPI.COMM_WORLD.barrier()

    @pytest.mark.skip(reason="FIXME reshape")
    def test_gather1(self):
        a = dt.reshape(dt.arange(0, 110, 1, dtype=dt.float64), [11, 10])
        b = dt.spmd.gather(a)
        c = np.sum(b)
        v = np.sum(np.reshape(np.arange(0, 110, 1, dtype=np.float64), (11, 10)))
        assert float(c) == v
        MPI.COMM_WORLD.barrier()

    def test_gather2(self):
        a = dt.arange(0, 110, 1, dtype=dt.float64)
        b = dt.spmd.gather(a)
        c = np.sum(b)
        v = np.sum(np.arange(0, 110, 1, dtype=np.float64))
        assert float(c) == v
        MPI.COMM_WORLD.barrier()

    @pytest.mark.skip(reason="FIXME reshape")
    def test_gather_strided1(self):
        a = dt.reshape(dt.arange(0, 110, 1, dtype=dt.float64), [11, 10])
        b = dt.spmd.gather(a[4:12:2, 1:11:3])
        c = np.sum(b)
        v = np.sum(
            np.reshape(np.arange(0, 110, 1, dtype=np.float64), (11, 10))[4:12:2, 1:11:3]
        )
        assert float(c) == v
        MPI.COMM_WORLD.barrier()

    def test_gather_strided2(self):
        a = dt.arange(0, 110, 1, dtype=dt.float64)
        b = dt.spmd.gather(a[34:82:2])
        c = np.sum(b)
        v = np.sum(np.arange(0, 110, 1, dtype=np.float64)[34:82:2])
        assert float(c) == v
        MPI.COMM_WORLD.barrier()

    @pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="FIXME multi-proc")
    def test_from_locals(self):
        npa = np.arange(1, 11, 2, dtype=np.int64)
        a = dt.spmd.from_locals(npa)
        x = 4711
        npa[0] = x
        assert int(a[0]) == x
