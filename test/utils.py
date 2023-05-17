import numpy
import ddptensor


def runAndCompare(func, do_gather=True):
    aa = func(ddptensor)
    a = ddptensor.spmd.gather(aa) if do_gather else aa
    b = func(numpy)
    if isinstance(b, numpy.ndarray):
        print(aa)
        print(a)
        print(b)
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
