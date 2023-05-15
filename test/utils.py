import numpy
import ddptensor


def runAndCompare(func, do_gather=True):
    a = func(ddptensor)
    if do_gather:
        a = ddptensor.spmd.gather(a)
    b = func(numpy)
    if isinstance(b, numpy.ndarray):
        return a.shape == b.shape and numpy.allclose(a, b, rtol=1e-8, atol=1e-8)
    return float(a) == float(b)


mpi_dtypes = [
    ddptensor.float64,
    ddptensor.float32,
    ddptensor.int64,
    ddptensor.uint64,
    ddptensor.int32,
    ddptensor.uint32,
    ddptensor.int8,
    ddptensor.uint8,
]
