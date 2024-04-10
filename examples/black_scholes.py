"""
The Black-Scholes program computes the price of a portfolio of
options using partial differential equations.
The entire computation performed by Black-Scholes is data-parallel
where each option can be priced independent of other options.

Examples:

    # run 1 million options, by default times 10 iterations
    python black_scholes.py -n 1000000

    # run 10 iterations of M16Gb problem size
    python black_scholes.py -i 10 -p M16Gb

    # use float32 data type
    python black_scholes.py -p M16Gb -d f32

    # MPI parallel run
    mpiexec -n 4 python black_scholes.py -p M16Gb

    # use numpy backend (serial only)
    python black_scholes.py -p M16Gb -b numpy

"""

import argparse
import os
import time as time_mod
from functools import partial

import numpy

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


def naive_erf(x):
    """
    Error function (erf) implementation

    Adapted from formula 7.1.26 in
    Abramowitz and Stegun, "Handbook of Mathematical Functions", 1965.
    """
    y = numpy.abs(x)

    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    t = 1.0 / (1.0 + p * y)
    f = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t
    return numpy.sign(x) * (1.0 - f * numpy.exp(-y * y))


def initialize(create_full, nopt, seed, dtype):
    """
    Initialize arrays.

    Input
    ---------
    np:
        Numpy-like array module
    nopt: int
        number of options
    seed: double
        random generator seed
    dtype:
        data type

    Output
    ------

    price, strike, t, call, put: array of shape (nopt, )
        data arrays
    rate, volatility: float
        rate and volatility parameters
    """

    # S0L = 10.0
    # S0H = 50.0
    # XL = 10.0
    # XH = 50.0
    # TL = 1.0
    # TH = 2.0
    RISK_FREE = 0.1
    VOLATILITY = 0.2

    # random initialization
    # random.seed(seed)
    # price = random.uniform(S0L, S0H, nopt)
    # strike = random.uniform(XL, XH, nopt)
    # t = random.uniform(TL, TH, nopt)

    # constant values
    price = create_full((nopt,), 49.81959369901096, dtype)
    strike = create_full((nopt,), 40.13264789835957, dtype)
    t = create_full((nopt,), 1.8994311692123782, dtype)
    # parameters
    rate = RISK_FREE
    volatility = VOLATILITY
    # output arrays
    call = create_full((nopt,), 0.0, dtype)  # 16.976097804669887
    put = create_full((nopt,), 0.0, dtype)  # 0.34645174725098116

    return (price, strike, t, rate, volatility, call, put)


def black_scholes(np, erf, nopt, price, strike, t, rate, volatility, call, put):
    """
    Evaluate the Black-Scholes equation.


    Input
    ---------
    np:
        Numpy-like array module
    erf:
        erf function implementation
    nopt: int
        number of options
    price, strike, t: array of shape (nopt, )
        vectors representing different components of portfolio
    rate, volatility: float
        scalars used for price computation

    Output
    -------
    call, put: array of shape (nopt, )
        output vectors
    """
    mr = -rate
    sig_sig_two = volatility * volatility * 2

    P = price
    S = strike
    T = t

    a = np.log(P / S)
    b = T * mr

    z = T * sig_sig_two
    c = 0.25 * z
    y = 1.0 / np.sqrt(z)

    w1 = (a - b + c) * y
    w2 = (a - b - c) * y

    d1 = 0.5 + 0.5 * erf(w1)
    d2 = 0.5 + 0.5 * erf(w2)

    Se = np.exp(b) * S

    r = P * d1 - Se * d2
    call[:] = r
    put[:] = r - P + Se


def run(nopt, backend, iterations, datatype):
    if backend == "sharpy":
        import sharpy as np
        from sharpy import fini, init, sync

        device = os.getenv("SHARPY_DEVICE", "")
        create_full = partial(np.full, device=device)
        erf = np.erf

        init(False)
    elif backend == "numpy":
        import numpy as np

        erf = naive_erf

        if comm is not None:
            assert (
                comm.Get_size() == 1
            ), "Numpy backend only supports serial execution."

        create_full = np.full
        fini = sync = lambda x=None: None
    else:
        raise ValueError(f'Unknown backend: "{backend}"')

    dtype = {
        "f32": np.float32,
        "f64": np.float64,
    }[datatype]

    seed = 777777

    info(f"Using backend: {backend}")
    info(f"Number of options: {nopt}")
    info(f"Datatype: {datatype}")

    # initialize
    args = initialize(create_full, nopt, seed, dtype)
    sync()

    def eval():
        tic = time_mod.perf_counter()
        black_scholes(np, erf, nopt, *args)
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
    perf_rate = nopt / t_med / 1e6  # million options per second
    init_overhead = t_warm - t_med
    if backend == "sharpy":
        info(f"Estimated initialization overhead: {init_overhead:.5f} s")
    info(f"Min.   duration: {t_min:.5f} s")
    info(f"Max.   duration: {t_max:.5f} s")
    info(f"Median duration: {t_med:.5f} s")
    info(f"Median rate: {perf_rate:.5f} Mopts/s")

    # verify
    call, put = args[-2], args[-1]
    expected_call = 16.976097804669887
    expected_put = 0.34645174725098116
    call_value = float(call[0])
    put_value = float(put[0])
    assert numpy.allclose(call_value, expected_call)
    assert numpy.allclose(put_value, expected_put)

    info("SUCCESS")
    fini()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Black-Scholes benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-p",
        "--preset",
        type=str,
        help="Use a preset problem size.",
        choices=["S", "M", "L", "M16Gb"],
        default="S",
    )
    parser.add_argument(
        "-n",
        "--noptions",
        type=int,
        default=-1,
        help="Number of of options to evaluate. If set, overrides --preset.",
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
    # dpbench preset sizes
    preset_dict = {
        "S": 524288,
        "M": 134217728,
        "L": 268435456,
        "M16Gb": 67108864,
    }
    nopt = args.noptions
    if nopt < 1:
        nopt = preset_dict[args.preset]
    run(
        nopt,
        args.backend,
        args.iterations,
        args.datatype,
    )
