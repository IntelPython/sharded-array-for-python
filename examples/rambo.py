"""
Rambo benchmark

Examples:

    # run 1000 iterations of 10 events and 100 outputs on sharpy backend
    python rambo.py -nevts 10 -nout 100 -b sharpy -i 1000

    # MPI parallel run
    mpiexec -n 3 python rambo.py -nevts 64 -nout 64 -b sharpy -i 1000

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


def sp_rambo(sp, sp_C1, sp_F1, sp_Q1, sp_output, nevts, nout):
    sp_C = 2.0 * sp_C1 - 1.0
    sp_S = sp.sqrt(1 - sp.square(sp_C))
    sp_F = 2.0 * sp.pi * sp_F1
    sp_Q = -sp.log(sp_Q1)

    sp_output[:, :, 0] = sp.reshape(sp_Q, (nevts, nout, 1))
    sp_output[:, :, 1] = sp.reshape(
        sp_Q * sp_S * sp.sin(sp_F), (nevts, nout, 1)
    )
    sp_output[:, :, 2] = sp.reshape(
        sp_Q * sp_S * sp.cos(sp_F), (nevts, nout, 1)
    )
    sp_output[:, :, 3] = sp.reshape(sp_Q * sp_C, (nevts, nout, 1))

    sharpy.sync()


def np_rambo(np, C1, F1, Q1, output, nevts, nout):
    C = 2.0 * C1 - 1.0
    S = np.sqrt(1 - np.square(C))
    F = 2.0 * np.pi * F1
    Q = -np.log(Q1)

    output[:, :, 0] = Q
    output[:, :, 1] = Q * S * np.sin(F)
    output[:, :, 2] = Q * S * np.cos(F)
    output[:, :, 3] = Q * C


def initialize(np, nevts, nout, seed, dtype):
    np.random.seed(seed)
    C1 = np.random.rand(nevts, nout)
    F1 = np.random.rand(nevts, nout)
    Q1 = np.random.rand(nevts, nout) * np.random.rand(nevts, nout)
    return (C1, F1, Q1, np.zeros((nevts, nout, 4), dtype))


def run(nevts, nout, backend, iterations, datatype):
    if backend == "sharpy":
        import sharpy as np
        from sharpy import fini, init, sync

        rambo = sp_rambo

        init(False)
    elif backend == "numpy":
        import numpy as np

        if comm is not None:
            assert (
                comm.Get_size() == 1
            ), "Numpy backend only supports serial execution."

        fini = sync = lambda x=None: None
        rambo = np_rambo
    else:
        raise ValueError(f'Unknown backend: "{backend}"')

    dtype = {
        "f32": np.float32,
        "f64": np.float64,
    }[datatype]

    info(f"Using backend: {backend}")
    info(f"Number of events: {nevts}")
    info(f"Number of outputs: {nout}")
    info(f"Datatype: {datatype}")

    seed = 7777
    C1, F1, Q1, output = initialize(np, nevts, nout, seed, dtype)
    sync()

    # verify
    if backend == "sharpy":
        sp_rambo(sharpy, C1, F1, Q1, output, nevts, nout)
        # sync() !! not work here?
        np_C1 = sharpy.to_numpy(C1)
        np_F1 = sharpy.to_numpy(F1)
        np_Q1 = sharpy.to_numpy(Q1)
        np_output = numpy.zeros((nevts, nout, 4))
        np_rambo(numpy, np_C1, np_F1, np_Q1, np_output, nevts, nout)
        assert numpy.allclose(sharpy.to_numpy(output), np_output)

    def eval():
        tic = time_mod.perf_counter()
        rambo(np, C1, F1, Q1, output, nevts, nout)
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
        description="Run rambo benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-nevts",
        "--num_events",
        type=int,
        default=10,
        help="Number of events to evaluate.",
    )
    parser.add_argument(
        "-nout",
        "--num_outputs",
        type=int,
        default=10,
        help="Number of outputs to evaluate.",
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
    nevts, nout = args.num_events, args.num_outputs
    run(
        nevts,
        nout,
        args.backend,
        args.iterations,
        args.datatype,
    )
