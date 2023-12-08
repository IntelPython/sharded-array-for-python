import numpy
import sharpy
from sharpy.numpy import fromfunction
from os import getenv

sharpy.fromfunction = fromfunction


def runAndCompare(func, do_gather=True):
    aa = func(sharpy)
    a = sharpy.spmd.gather(aa) if do_gather else aa
    b = func(numpy)
    if isinstance(b, numpy.ndarray):
        return a.shape == b.shape and numpy.allclose(a, b, rtol=1e-8, atol=1e-8)
    return float(a) == float(b)


mpi_dtypes = [
    sharpy.float32,
    sharpy.int32,
]

on_gpu = getenv("SHARPY_USE_GPU", False)

if not on_gpu:
    mpi_dtypes += [
        sharpy.float64,
        sharpy.int64,
        sharpy.uint64,
        sharpy.uint32,
        sharpy.int8,
        sharpy.uint8,
    ]
