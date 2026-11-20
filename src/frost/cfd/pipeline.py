"""The closure on a second device, one outer step behind (mechanism M2 for CFD; test 13).

The C2 pipeline of MD (``frost.async_teacher``), for a learned closure. Interval
n integrates k solver steps on the solver device using what the closure device
computed from the state at the start of interval n - 1. Meanwhile the closure
device evaluates the state at the start of interval n:

    held_0 = refresh(w_0)                 (synchronous, as MD's first kick)
    for n: nxt = refresh(w_n)  on device 1, concurrent with
           w_{n+1} = inner(w_n, held_n)  on device 0;  held_{n+1} = nxt

JAX dispatches asynchronously and tracks the cross-device dependencies, so the
loop is plain Python. The solver device waits only when a refresh takes longer
than k solver steps. ``lag=0`` runs the same loop but waits for each refresh
(no overlap), which separates transfer and dispatch cost from the pipelining.

Unlike MD, where a lagged residual anti-damps oscillators, a stale dissipative
closure stays stable. Test 5 measured lag L at about the error of hold 2L, so
the depth is limited by the accuracy budget.
"""

from __future__ import annotations

from functools import partial

import jax

from frost.cfd import schedule
from frost.cfd.spectral import Grid


@partial(jax.jit, static_argnames=("grid", "mode", "cs"))
def refresh(grid: Grid, model, w_hat, mode: str, cs: float):
    return schedule._refresh(grid, model, w_hat, mode, cs)


@partial(jax.jit, static_argnames=("grid", "mode", "k", "dt", "cs"))
def inner(grid: Grid, w_hat, held, mode: str, k: int, dt: float, cs: float):
    def body(carry, _):
        return schedule._inner(grid, carry[0], carry[1], mode, dt, cs), None

    (w_hat, _), _ = jax.lax.scan(body, (w_hat, held), None, length=k)
    return w_hat


def run(grid: Grid, model, w_hat, *, mode: str, k: int, dt: float, n_outer: int, lag: int,
        solver_device, closure_device, cs: float = 0.17, record_every: int | None = None):
    """Returns (final state, records every ``record_every`` solver steps)."""
    if lag not in (0, 1):
        raise ValueError("lag 0 or 1")
    if record_every is not None and record_every % k:
        raise ValueError("record_every must be a multiple of k")
    model_c = jax.device_put(model, closure_device)
    w = jax.device_put(w_hat, solver_device)
    held = jax.device_put(refresh(grid, model_c, jax.device_put(w, closure_device), mode, cs),
                          solver_device)
    records = []
    for n in range(n_outer):
        if lag:
            nxt = refresh(grid, model_c, jax.device_put(w, closure_device), mode, cs)
            w = inner(grid, w, held, mode, k, dt, cs)
            held = jax.device_put(nxt, solver_device)
        else:
            if n:
                held = jax.device_put(
                    refresh(grid, model_c, jax.device_put(w, closure_device), mode, cs),
                    solver_device)
            w = inner(grid, w, held, mode, k, dt, cs)
        if record_every is not None and ((n + 1) * k) % record_every == 0:
            records.append(w)
    return w, records
