from mpi4py import MPI
import ddptensor as dt
a = dt.ones((8,8), dt.float64)
MPI.COMM_WORLD.barrier()
b = dt.spmd.get_slice(a, (slice(1, 4+MPI.COMM_WORLD.rank), slice(2, 4+MPI.COMM_WORLD.rank)))
print(b, type(b), flush=True)
l = dt.spmd.get_local(a)
print(l, l.dtype, flush=True)
l[0,0]=33.3
print(l[0,0], float(a[0:1,0:1]))
print(a, l)
aa = dt.arange(0, 64, 1, dtype=dt.int64)
aaa = dt.reshape(aa, [8,8])
print(aaa)
c = aaa[1:8:2, 1:8:4]
lc = dt.spmd.get_local(c)
print(c, lc)
#print(type(b), b.shape, float(b[1,1]))
print("done", flush=True)
dt.fini()
