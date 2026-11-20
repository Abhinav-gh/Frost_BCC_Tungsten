"""The 3D closure (``frost.cfd.split3d``) on a second device, one hold behind (M2 for W3).

``pipeline.py`` for the 3D split operator. Hold block n integrates k solver steps on the
solver device with the correction the closure device computed from the state at the start
of block n - 1, while the closure device evaluates the state at the start of block n:

    held_0 = correction(u_0)                     (synchronous)
    for n: nxt = correction(u_n)   on device 1, concurrent with
           u_{n+1} = inner(u_n, held_n)  on device 0;  held_{n+1} = nxt

JAX dispatches asynchronously and orders the cross-device copies, so the loop is plain
Python and nothing is patched. The solver device waits only when a correction takes longer
than k solver steps; at 32^3 and batch 16 the network costs ~25.6 ms against ~2.2 ms per
solver step, so k >= 12 hides it. ``lag=0`` runs the same loop but refreshes from the
current state before each block (the ordinary hold, serialised), which separates transfer
and dispatch cost from the overlap. A lagged correction is one hold staler than a held one,
so its statistics are certified like any other schedule (``experiments/e10_cfd/stats3d.py``).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax

from frost.cfd import split3d
from frost.cfd.spectral3d import Grid3


@partial(jax.jit, static_argnames=("grid",))
def refresh(grid: Grid3, model: dict, u_hat):
    return split3d.correction(grid, model, u_hat)


@partial(jax.jit, static_argnames=("grid", "k", "dt", "base", "physics", "substeps"))
def inner(grid: Grid3, u_hat, c_hat, k: int, dt: float, base: tuple[str, float], physics: str = "hydro",
          substeps: int = 1):
    return jax.lax.fori_loop(0, k, lambda _, x: split3d.split_step(grid, x, c_hat, dt, base, physics, substeps),
                             u_hat)


@jax.jit
def _extrapolate(c_new, c_old):
    return 2.0 * c_new - c_old


def run(grid: Grid3, model: dict, u_hat, *, k: int, dt: float, n_outer: int, lag: int,
        solver_device, closure_device, observe: Callable | None = None, extrapolate: bool = False):
    """Returns (final state, [observe(state) after each hold block]) on the solver device.

    ``observe`` should be jitted and return something small; it is dispatched, not waited on.
    ``extrapolate`` (lag 1 only): block n uses 2 c(u_{n-1}) - c(u_{n-2}), the lagged correction carried
    one hold forward linearly, instead of c(u_{n-1}) as it is: the same overlap, a staleness that is
    second order in the correction's change instead of first (a dissipative flow's correction varies
    smoothly on the eddy time; MD's oscillatory residual does not, which is why C2 holds instead).
    """
    if lag not in (0, 1):
        raise ValueError("lag 0 or 1")
    base, phys = split3d.base_of(model), split3d.physics_of(model)
    model_c = jax.device_put(model, closure_device)
    u = jax.device_put(u_hat, solver_device)
    held = jax.device_put(refresh(grid, model_c, jax.device_put(u, closure_device)), solver_device)
    last = held
    out = []
    for n in range(n_outer):
        if lag:
            nxt = refresh(grid, model_c, jax.device_put(u, closure_device))
            u = inner(grid, u, held, k, dt, base, *phys)
            arrived = jax.device_put(nxt, solver_device)
            held = _extrapolate(arrived, last) if extrapolate and n > 0 else arrived
            last = arrived
        else:
            u = inner(grid, u, held, k, dt, base, *phys)
            held = jax.device_put(refresh(grid, model_c, jax.device_put(u, closure_device)), solver_device)
        if observe is not None:
            out.append(observe(u))
    return u, out
