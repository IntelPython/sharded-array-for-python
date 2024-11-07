"""
Linear wave equation benchmark

Usage:

Verify solution with 128x128 problem size

.. code-block::

    python shallow_water.py

Run a performance test with 1024x1024 problem size.
Runs a fixed number of steps with a small time step.

.. code-block::

    python shallow_water.py -n 1024 -t

Run with numpy backend

.. code-block::

    python shallow_water.py -b numpy ...

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
    g = 9.81
    coriolis = 10.0

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
    F_shape = (nx + 1, ny + 1)
    sync()
    x_t_2d = xmin + ind_arr(T_shape, True) * dx + dx / 2
    y_t_2d = ymin + ind_arr(T_shape) * dy + dy / 2

    x_u_2d = xmin + ind_arr(U_shape, True) * dx
    y_u_2d = ymin + ind_arr(U_shape) * dy + dy / 2

    x_v_2d = xmin + ind_arr(V_shape, True) * dx + dx / 2
    y_v_2d = ymin + ind_arr(V_shape) * dy
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

    # potential vorticity
    q = create_full(F_shape, 0.0, dtype)

    # bathymetry
    h = create_full(T_shape, 0.0, dtype)

    hu = create_full(U_shape, 0.0, dtype)
    hv = create_full(V_shape, 0.0, dtype)

    dudy = create_full(F_shape, 0.0, dtype)
    dvdx = create_full(F_shape, 0.0, dtype)

    # vector invariant form
    H_at_f = create_full(F_shape, 0.0, dtype)

    # auxiliary variables for RK time integration
    e1 = create_full(T_shape, 0.0, dtype)
    u1 = create_full(U_shape, 0.0, dtype)
    v1 = create_full(V_shape, 0.0, dtype)
    e2 = create_full(T_shape, 0.0, dtype)
    u2 = create_full(U_shape, 0.0, dtype)
    v2 = create_full(V_shape, 0.0, dtype)

    def exact_solution(t, x_t_2d, y_t_2d, x_u_2d, y_u_2d, x_v_2d, y_v_2d):
        """
        Exact solution for a stationary geostrophic gyre.
        """
        f = coriolis
        amp = 0.02
        x0 = 0.0
        y0 = 0.0
        sigma = 0.4
        x = x_t_2d - x0
        y = y_t_2d - y0
        x_u = x_u_2d - x0
        y_u = y_u_2d - y0
        x_v = x_v_2d - x0
        y_v = y_v_2d - y0
        elev = amp * np.exp(-1.0 * (x**2.0 + y**2.0) / sigma**2.0)
        elev_u = amp * np.exp(-1.0 * (x_u**2.0 + y_u**2.0) / sigma**2.0)
        elev_v = amp * np.exp(-1.0 * (x_v**2.0 + y_v**2.0) / sigma**2.0)
        u = g / f * 2.0 * y_u / sigma**2.0 * elev_u
        v = -1.0 * g / f * 2.0 * x_v / sigma**2.0 * elev_v
        return u, v, elev

    def bathymetry(x_t_2d, y_t_2d, lx, ly):
        """
        Water depth at rest
        """
        bath = 1.0
        return bath * create_full(T_shape, 1.0, dtype)

    # set bathymetry
    h[:, :] = bathymetry(x_t_2d, y_t_2d, lx, ly)
    # steady state potential energy
    pe_offset = 0.5 * g * float(np.sum(h**2.0, all_axes)) / nx / ny

    # compute time step
    alpha = 0.5
    h_max = float(np.max(h, all_axes))
    c = (g * h_max) ** 0.5
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

        # total depth
        H = e + h

        # volume flux divergence -div(H u)
        hu[1:-1, :] = 0.5 * (H[:-1, :] + H[1:, :]) * u[1:-1, :]
        hv[:, 1:-1] = 0.5 * (H[:, :-1] + H[:, 1:]) * v[:, 1:-1]

        dedt = -1.0 * (
            (hu[1:, :] - hu[:-1, :]) / dx + (hv[:, 1:] - hv[:, :-1]) / dy
        )

        # total depth at F points
        H_at_f[1:-1, 1:-1] = 0.25 * (
            H[1:, 1:] + H[:-1, 1:] + H[1:, :-1] + H[:-1, :-1]
        )
        H_at_f[0, 1:-1] = 0.5 * (H[0, 1:] + H[0, :-1])
        H_at_f[-1, 1:-1] = 0.5 * (H[-1, 1:] + H[-1, :-1])
        H_at_f[1:-1, 0] = 0.5 * (H[1:, 0] + H[:-1, 0])
        H_at_f[1:-1, -1] = 0.5 * (H[1:, -1] + H[:-1, -1])
        H_at_f[0, 0] = H[0, 0]
        H_at_f[0, -1] = H[0, -1]
        H_at_f[-1, 0] = H[-1, 0]
        H_at_f[-1, -1] = H[-1, -1]

        # potential vorticity
        dudy[:, 1:-1] = (u[:, 1:] - u[:, :-1]) / dy
        dvdx[1:-1, :] = (v[1:, :] - v[:-1, :]) / dx
        q[:, :] = (coriolis - dudy + dvdx) / H_at_f

        # Advection of potential vorticity, Arakawa and Hsu (1990)
        # Define alpha, beta, gamma, delta for each cell in T points
        w = 1.0 / 12
        q_a = w * (q[:-1, 1:] + q[:-1, :-1] + q[1:, 1:])
        q_b = w * (q[1:, 1:] + q[1:, :-1] + q[:-1, 1:])
        q_g = w * (q[1:, :-1] + q[1:, 1:] + q[:-1, :-1])
        q_d = w * (q[:-1, :-1] + q[:-1, 1:] + q[1:, :-1])

        # kinetic energy
        u2 = u * u
        v2 = v * v
        u2_at_t = 0.5 * (u2[1:, :] + u2[:-1, :])
        v2_at_t = 0.5 * (v2[:, 1:] + v2[:, :-1])
        ke = 0.5 * (u2_at_t + v2_at_t)

        dudt = (
            # pressure gradient -g grad(elev)
            -g * (e[1:, :] - e[:-1, :]) / dx
            # kinetic energy gradient
            - (ke[1:, :] - ke[:-1, :]) / dx
            # potential vorticity advection terms
            + q_a[1:, :] * hv[1:, 1:]
            + q_b[:-1, :] * hv[:-1, 1:]
            + q_g[:-1, :] * hv[:-1, :-1]
            + q_d[1:, :] * hv[1:, :-1]
        )
        dvdt = (
            # pressure gradient -g grad(elev)
            -g * (e[:, 1:] - e[:, :-1]) / dy
            # kinetic energy gradient
            - (ke[:, 1:] - ke[:, :-1]) / dy
            # potential vorticity advection terms
            - q_g[:, 1:] * hu[1:, 1:]
            - q_d[:, 1:] * hu[:-1, 1:]
            - q_a[:, :-1] * hu[:-1, :-1]
            - q_b[:, :-1] * hu[1:, :-1]
        )

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
    u0, v0, e0 = exact_solution(
        0, x_t_2d, y_t_2d, x_u_2d, y_u_2d, x_v_2d, y_v_2d
    )
    e[:, :] = e0
    u[:, :] = u0
    v[:, :] = v0

    t = 0
    i_export = 0
    next_t_export = 0
    initial_v = None
    initial_e = None
    tic = time_mod.perf_counter()
    block_tic = 0
    for i in range(nt + 1):
        sync()
        t = i * dt

        if t >= next_t_export - 1e-8:
            _elev_max = np.max(e, all_axes)
            _u_max = np.max(u, all_axes)
            _q_max = np.max(q, all_axes)
            _total_v = np.sum(e + h, all_axes)

            # potential energy
            _pe = 0.5 * g * (e + h) * (e - h) + pe_offset
            _total_pe = np.sum(_pe, all_axes)

            # kinetic energy
            u2 = u * u
            v2 = v * v
            u2_at_t = 0.5 * (u2[1:, :] + u2[:-1, :])
            v2_at_t = 0.5 * (v2[:, 1:] + v2[:, :-1])
            _ke = 0.5 * (u2_at_t + v2_at_t) * (e + h)
            _total_ke = np.sum(_ke, all_axes)

            total_pe = float(_total_pe) * dx * dy
            total_ke = float(_total_ke) * dx * dy
            total_e = total_ke + total_pe
            elev_max = float(_elev_max)
            u_max = float(_u_max)
            q_max = float(_q_max)
            total_v = float(_total_v) * dx * dy

            if i_export == 0:
                initial_v = total_v
                initial_e = total_e
                tcpu_str = ""
            else:
                block_duration = time_mod.perf_counter() - block_tic
                tcpu_str = f" Tcpu={block_duration:.3} s"

            diff_v = total_v - initial_v
            diff_e = total_e - initial_e

            info(
                f"{i_export:2d} {i:4d} {t:.3f} elev={elev_max:7.5f} "
                f"u={u_max:7.5f} q={q_max:8.5f} dV={diff_v: 6.3e} "
                f"PE={total_pe:7.2e} KE={total_ke:7.2e} dE={diff_e: 6.3e}"
                + tcpu_str
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

    e_exact = exact_solution(t, x_t_2d, y_t_2d, x_u_2d, y_u_2d, x_v_2d, y_v_2d)[
        2
    ]
    err2 = (e_exact - e) * (e_exact - e) * dx * dy / lx / ly
    err_L2 = math.sqrt(float(np.sum(err2, all_axes)))
    info(f"L2 error: {err_L2:7.15e}")

    if nx < 128 or ny < 128:
        info("Skipping correctness test due to small problem size.")
    elif not benchmark_mode:
        tolerance_ene = 1e-7 if datatype == "f32" else 1e-9
        assert (
            diff_e < tolerance_ene
        ), f"Energy error exceeds tolerance: {diff_e} > {tolerance_ene}"
        if nx == 128 and ny == 128:
            if datatype == "f32":
                assert numpy.allclose(
                    err_L2, 4.3127859e-05, rtol=1e-5
                ), "L2 error does not match"
            else:
                assert numpy.allclose(
                    err_L2, 4.315799035627906e-05
                ), "L2 error does not match"
        else:
            tolerance_l2 = 1e-4
            assert (
                err_L2 < tolerance_l2
            ), f"L2 error exceeds tolerance: {err_L2} > {tolerance_l2}"
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
