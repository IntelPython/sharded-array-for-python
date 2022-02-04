from mpi4py import MPI
import ddptensor as dt
a = dt.ones((8,8))
MPI.COMM_WORLD.barrier()
b = a.get_slice((slice(1, 3+MPI.COMM_WORLD.rank), slice(2, 4+MPI.COMM_WORLD.rank)))
print(type(b), b.shape, float(b[1,1]))
print("done")
dt.fini()
