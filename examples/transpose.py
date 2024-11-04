"""
Transpose benchmark

    Matrix transpose benchmark for sharpy and numpy backends.

Examples:

    # Run 1000 iterations of 1000*1000 matrix on sharpy backend
    python transpose.py -r 10 -c 1000 -b sharpy -i 1000

    # MPI parallel run
    mpiexec -n 3 python transpose.py -r 1000 -c 1000 -b sharpy -i 1000

"""

import argparse
import time as time_mod

import numpy

import sharpy

try:
    import mpi4py

    mpi4py.rc.finalize = False
    from mpi4py import MPI

    comm_rank = MPI.COMM_WORLD.Get_rank()
    comm = MPI.COMM_WORLD
except ImportError:
    comm_rank = 0
    comm = None


def info(s):
    if comm_rank == 0:
        print(s)


def sp_transpose(arr):
    brr = sharpy.permute_dims(arr, [1, 0])
    return brr


def np_transpose(arr):
    brr = arr.transpose()
    return brr.copy()


def initialize(np, row, col, dtype):
    arr = np.arange(0, row * col, 1, dtype=dtype)
    return np.reshape(arr, (row, col))


def run(row, col, backend, iterations, datatype):
    if backend == "sharpy":
        import sharpy as np
        from sharpy import fini, init, sync

        transpose = sp_transpose

        init(False)
    elif backend == "numpy":
        import numpy as np

        if comm is not None:
            assert (
                comm.Get_size() == 1
            ), "Numpy backend only supports serial execution."

        fini = sync = lambda x=None: None
        transpose = np_transpose
    else:
        raise ValueError(f'Unknown backend: "{backend}"')

    dtype = {
        "f32": np.float32,
        "f64": np.float64,
    }[datatype]

    info(f"Using backend: {backend}")
    info(f"Number of row: {row}")
    info(f"Number of column: {col}")
    info(f"Datatype: {datatype}")

    arr = initialize(np, row, col, dtype)
    sync()

    # verify
    if backend == "sharpy":
        brr = sp_transpose(arr)
        crr = np_transpose(sharpy.to_numpy(arr))
        assert numpy.allclose(sharpy.to_numpy(brr), crr)

    def eval():
        tic = time_mod.perf_counter()
        transpose(arr)
        sync()
        toc = time_mod.perf_counter()
        return toc - tic

    # warm-up run
    t_warm = eval()

    # evaluate
    info(f"Running {iterations} iterations")
    time_list = []
    for i in range(iterations):
        time_list.append(eval())

    # get max time over mpi ranks
    if comm is not None:
        t_warm = comm.allreduce(t_warm, MPI.MAX)
        time_list = comm.allreduce(time_list, MPI.MAX)

    t_min = numpy.min(time_list)
    t_max = numpy.max(time_list)
    t_med = numpy.median(time_list)
    init_overhead = t_warm - t_med
    if backend == "sharpy":
        info(f"Estimated initialization overhead: {init_overhead:.5f} s")
    info(f"Min.   duration: {t_min:.5f} s")
    info(f"Max.   duration: {t_max:.5f} s")
    info(f"Median duration: {t_med:.5f} s")

    fini()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run transpose benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-r",
        "--row",
        type=int,
        default=10000,
        help="Number of row.",
    )
    parser.add_argument(
        "-c",
        "--column",
        type=int,
        default=10000,
        help="Number of column.",
    )

    parser.add_argument(
        "-b",
        "--backend",
        type=str,
        default="sharpy",
        choices=["sharpy", "numpy"],
        help="Backend to use.",
    )

    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations to run.",
    )

    parser.add_argument(
        "-d",
        "--datatype",
        type=str,
        default="f64",
        choices=["f32", "f64"],
        help="Datatype for model state variables",
    )

    args = parser.parse_args()
    run(
        args.row,
        args.column,
        args.backend,
        args.iterations,
        args.datatype,
    )
