from mpi4py import MPI
import ddptensor as dt
a = dt.ones((8,8), dt.float64)
MPI.COMM_WORLD.barrier()
b = dt.spmd.get_slice(a, (slice(1, 4+MPI.COMM_WORLD.rank), slice(2, 4+MPI.COMM_WORLD.rank)))
print(b, type(b))
#print(type(b), b.shape, float(b[1,1]))
print("done")
dt.fini()
