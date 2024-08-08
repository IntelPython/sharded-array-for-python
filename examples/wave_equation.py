"""
Linear wave equation benchmark

Usage:

Verify solution with 128x128 problem size

.. code-block::

    python wave_equation.py

Run a performance test with 1024x1024 problem size.
Runs a fixed number of steps with a small time step.

.. code-block::

    python wave_equation.py -n 1024 -t

Run with numpy backend

.. code-block::

    python wave_equation.py -b numpy ...

"""

import argparse
import math
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


def run(n, backend, datatype, benchmark_mode):
    if backend == "sharpy":
        import sharpy as np
        from sharpy import fini, init, sync

        device = os.getenv("SHARPY_DEVICE", "")
        create_full = partial(np.full, device=device)

        def transpose(a):
            return np.permute_dims(a, [1, 0])

        all_axes = [0, 1]
        init(False)

    elif backend == "numpy":
        import numpy as np

        if comm is not None:
            assert (
                comm.Get_size() == 1
            ), "Numpy backend only supports serial execution."

        create_full = np.full
        transpose = np.transpose

        fini = sync = lambda x=None: None
        all_axes = None
    else:
        raise ValueError(f'Unknown backend: "{backend}"')

    info(f"Using backend: {backend}")

    dtype = {
        "f64": np.float64,
        "f32": np.float32,
    }[datatype]
    info(f"Datatype: {datatype}")

    # constants
    h = 1.0
    g = 9.81

    # domain extent
    xmin = -1
    xmax = 1
    ymin = -1
    ymax = 1
    lx = xmax - xmin
    ly = ymax - ymin

    # grid resolution
    nx = n
    ny = n
    # grid spacing
    dx = lx / nx
    dy = lx / ny

    # export interval
    t_export = 0.02
    t_end = 1.0

    def ind_arr(shape, columns=False):
        """Construct an (nx, ny) array where each row/col is an arange"""
        nx, ny = shape
        if columns:
            ind = np.arange(0, nx * ny, 1, dtype=np.int32) % nx
            ind = transpose(np.reshape(ind, (ny, nx)))
        else:
            ind = np.arange(0, nx * ny, 1, dtype=np.int32) % ny
            ind = np.reshape(ind, (nx, ny))
        return ind.astype(dtype)

    # coordinate arrays
    T_shape = (nx, ny)
    U_shape = (nx + 1, ny)
    V_shape = (nx, ny + 1)
    sync()
    x_t_2d = xmin + ind_arr(T_shape, True) * dx + dx / 2
    y_t_2d = ymin + ind_arr(T_shape) * dy + dy / 2
    sync()

    dofs_T = int(numpy.prod(numpy.asarray(T_shape)))
    dofs_U = int(numpy.prod(numpy.asarray(U_shape)))
    dofs_V = int(numpy.prod(numpy.asarray(V_shape)))

    info(f"Grid size: {nx} x {ny}")
    info(f"Elevation DOFs: {dofs_T}")
    info(f"Velocity  DOFs: {dofs_U + dofs_V}")
    info(f"Total     DOFs: {dofs_T + dofs_U + dofs_V}")

    # prognostic variables: elevation, (u, v) velocity
    e = create_full(T_shape, 0.0, dtype)
    u = create_full(U_shape, 0.0, dtype)
    v = create_full(V_shape, 0.0, dtype)

    # auxiliary variables for RK time integration
    e1 = create_full(T_shape, 0.0, dtype)
    u1 = create_full(U_shape, 0.0, dtype)
    v1 = create_full(V_shape, 0.0, dtype)
    e2 = create_full(T_shape, 0.0, dtype)
    u2 = create_full(U_shape, 0.0, dtype)
    v2 = create_full(V_shape, 0.0, dtype)

    sync()

    def exact_elev(t, x_t_2d, y_t_2d, lx, ly):
        """
        Exact solution for elevation field.

        Returns time-dependent elevation of a 2D standing wave in a rectangular
        domain.
        """
        amp = 0.5
        c = (g * h) ** 0.5
        n = 1
        sol_x = np.cos(2 * n * math.pi * x_t_2d / lx)
        m = 1
        sol_y = np.cos(2 * m * math.pi * y_t_2d / ly)
        omega = c * math.pi * ((n / lx) ** 2 + (m / ly) ** 2) ** 0.5
        # NOTE sharpy fails with scalar computation
        sol_t = numpy.cos(2 * omega * t)
        return amp * sol_x * sol_y * sol_t

    # compute time step
    alpha = 0.5
    c = (g * h) ** 0.5
    dt = alpha * dx / c
    dt = t_export / int(math.ceil(t_export / dt))
    nt = int(math.ceil(t_end / dt))
    if benchmark_mode:
        dt = 1e-5
        nt = 100
        t_export = dt * 25

    info(f"Time step: {dt} s")
    info(f"Total run time: {t_end} s, {nt} time steps")

    sync()

    def rhs(u, v, e):
        """
        Evaluate right hand side of the equations
        """
        # sign convention: positive on rhs

        # pressure gradient -g grad(elev)
        dudt = -g * (e[1:, :] - e[:-1, :]) / dx
        dvdt = -g * (e[:, 1:] - e[:, :-1]) / dy

        # velocity divergence -h div(u)
        dedt = -h * ((u[1:, :] - u[:-1, :]) / dx + (v[:, 1:] - v[:, :-1]) / dy)

        return dudt, dvdt, dedt

    def step(u, v, e, u1, v1, e1, u2, v2, e2):
        """
        Execute one SSPRK(3,3) time step
        """
        dudt, dvdt, dedt = rhs(u, v, e)
        u1[1:-1, :] = u[1:-1, :] + dt * dudt
        v1[:, 1:-1] = v[:, 1:-1] + dt * dvdt
        e1[:, :] = e[:, :] + dt * dedt

        dudt, dvdt, dedt = rhs(u1, v1, e1)
        u2[1:-1, :] = 0.75 * u[1:-1, :] + 0.25 * (u1[1:-1, :] + dt * dudt)
        v2[:, 1:-1] = 0.75 * v[:, 1:-1] + 0.25 * (v1[:, 1:-1] + dt * dvdt)
        e2[:, :] = 0.75 * e[:, :] + 0.25 * (e1[:, :] + dt * dedt)

        dudt, dvdt, dedt = rhs(u2, v2, e2)
        u[1:-1, :] = u[1:-1, :] / 3.0 + 2.0 / 3.0 * (u2[1:-1, :] + dt * dudt)
        v[:, 1:-1] = v[:, 1:-1] / 3.0 + 2.0 / 3.0 * (v2[:, 1:-1] + dt * dvdt)
        e[:, :] = e[:, :] / 3.0 + 2.0 / 3.0 * (e2[:, :] + dt * dedt)

    # warm up jit cache
    step(u, v, e, u1, v1, e1, u2, v2, e2)
    sync()

    # initial solution
    e[:, :] = exact_elev(0.0, x_t_2d, y_t_2d, lx, ly).to_device(device)
    u[:, :] = create_full(U_shape, 0.0, dtype)
    v[:, :] = create_full(V_shape, 0.0, dtype)
    sync()

    t = 0
    i_export = 0
    next_t_export = 0
    initial_v = None
    tic = time_mod.perf_counter()
    block_tic = 0
    for i in range(nt + 1):
        sync()
        t = i * dt

        if t >= next_t_export - 1e-8:
            if device:
                # FIXME gpu.memcpy to host requires identity layout
                # FIXME reduction on gpu
                # e_host = e.to_device()
                # u_host = u.to_device()
                # h_host = h.to_device()
                # _elev_max = np.max(e_host, all_axes)
                # _u_max = np.max(u_host, all_axes)
                # _total_v = np.sum(e_host + h, all_axes)
                _elev_max = 0
                _u_max = 0
                _total_v = 0
            else:
                _elev_max = np.max(e, all_axes)
                _u_max = np.max(u, all_axes)
                _total_v = np.sum(e + h, all_axes)

            elev_max = float(_elev_max)
            u_max = float(_u_max)
            total_v = float(_total_v) * dx * dy

            if i_export == 0:
                initial_v = total_v
                tcpu_str = ""
            else:
                block_duration = time_mod.perf_counter() - block_tic
                tcpu_str = f"Tcpu={block_duration:.3} s"

            diff_v = total_v - initial_v

            info(
                f"{i_export:2d} {i:4d} {t:.3f} elev={elev_max:7.5f} "
                f"u={u_max:7.5f} dV={diff_v: 6.3e} " + tcpu_str
            )
            if not benchmark_mode and (
                elev_max > 1e3 or not math.isfinite(elev_max)
            ):
                raise ValueError(f"Invalid elevation value: {elev_max}")
            i_export += 1
            next_t_export = i_export * t_export
            sync()
            block_tic = time_mod.perf_counter()

        step(u, v, e, u1, v1, e1, u2, v2, e2)

    sync()

    duration = time_mod.perf_counter() - tic
    info(f"Duration: {duration:.2f} s")

    if device:
        # FIXME gpu.memcpy to host requires identity layout
        # FIXME reduction on gpu
        # err2_host = err2.to_device()
        # err_L2 = math.sqrt(float(np.sum(err2_host, all_axes)))
        err_L2 = 0
    else:
        e_exact = exact_elev(t, x_t_2d, y_t_2d, lx, ly)
        err2 = (e_exact - e) * (e_exact - e) * dx * dy / lx / ly
        err_L2 = math.sqrt(float(np.sum(err2, all_axes)))
    info(f"L2 error: {err_L2:7.5e}")

    if nx == 128 and ny == 128 and not benchmark_mode and not device:
        if datatype == "f32":
            assert numpy.allclose(err_L2, 7.2235471e-03, rtol=1e-4)
        else:
            assert numpy.allclose(err_L2, 7.224068445111e-03)
        info("SUCCESS")

    fini()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run wave equation benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n",
        "--resolution",
        type=int,
        default=128,
        help="Number of grid cells in x and y direction.",
    )
    parser.add_argument(
        "-t",
        "--benchmark-mode",
        action="store_true",
        help="Run a fixed number of time steps.",
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
        "-d",
        "--datatype",
        type=str,
        default="f64",
        choices=["f32", "f64"],
        help="Datatype for model state variables",
    )
    args = parser.parse_args()
    run(
        args.resolution,
        args.backend,
        args.datatype,
        args.benchmark_mode,
    )
