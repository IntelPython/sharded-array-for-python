import numpy as np
from mpi4py import MPI
import ddptensor as dt
from conftest import ddpt_cw
import pytest

@pytest.mark.skipif(ddpt_cw, reason="Only applicable to SPMD mode")
class TestSPMD:
    def test_get_slice(self):
        a = dt.ones((2+MPI.COMM_WORLD.size*2, 2+MPI.COMM_WORLD.size*2), dt.float64)
        MPI.COMM_WORLD.barrier()
        b = dt.spmd.get_slice(a, (slice(MPI.COMM_WORLD.rank, MPI.COMM_WORLD.rank*2), slice(1, 2+MPI.COMM_WORLD.rank)))
        c = np.sum(b)
        v = MPI.COMM_WORLD.rank*(1+MPI.COMM_WORLD.rank)
        assert c == v

    def test_get_local1(self):
        a = dt.ones((32, 32), dt.float64)
        l = dt.spmd.get_local(a)
        l[0,0] = 0
        c = dt.sum(a, [0,1])
        v = 32*32-MPI.COMM_WORLD.size
        assert float(c) == v

    def test_get_local2(self):
        a = dt.ones((32, 32), dt.float64)
        b = a[0:32:2, 0:32:2]
        l = dt.spmd.get_local(a)
        l[0,0] = 0
        c = dt.sum(b, [0,1])
        v = 16*16-MPI.COMM_WORLD.size
        assert float(c) == v
