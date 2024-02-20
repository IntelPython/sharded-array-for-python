import numpy
import sharpy
from sharpy.numpy import fromfunction
from os import getenv

sharpy.fromfunction = fromfunction

device = getenv("SHARPY_USE_GPU", "")


def runAndCompare(func, do_gather=True):
    aa = func(sharpy, device=device)
    a = sharpy.spmd.gather(aa) if do_gather else aa
    b = func(numpy)
    if isinstance(b, numpy.ndarray):
        return a.shape == b.shape and numpy.allclose(a, b, rtol=1e-8, atol=1e-8)
    return float(a) == float(b)


mpi_dtypes = [
    sharpy.float32,
    sharpy.int32,
    sharpy.uint32,
]

if len(device) == 0:
    mpi_dtypes += [
        sharpy.float64,
        sharpy.int64,
        sharpy.uint64,
        sharpy.int8,
        sharpy.uint8,
    ]
