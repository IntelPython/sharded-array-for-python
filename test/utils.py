import numpy
import ddptensor
from ddptensor.numpy import fromfunction
from os import getenv

ddptensor.fromfunction = fromfunction


def runAndCompare(func, do_gather=True):
    aa = func(ddptensor)
    a = ddptensor.spmd.gather(aa) if do_gather else aa
    b = func(numpy)
    if isinstance(b, numpy.ndarray):
        return a.shape == b.shape and numpy.allclose(a, b, rtol=1e-8, atol=1e-8)
    return float(a) == float(b)


mpi_dtypes = [
    ddptensor.float32,
    ddptensor.int32,
]

on_gpu = getenv("DDPT_USE_GPU", False)

if not on_gpu:
    mpi_dtypes += [
        ddptensor.float64,
        ddptensor.int64,
        ddptensor.uint64,
        ddptensor.uint32,
        ddptensor.int8,
        ddptensor.uint8,
    ]
