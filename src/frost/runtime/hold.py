"""CFD / MHD backend for the runtime: a learned closure refreshed every k steps of the solver.

The operator is ``frost.cfd.split3d.correction``, value-consumed (added to the tendency), so the
runtime may hold its output (``reuse.check_hold(VALUE, "output")``). One interval is one jitted
call: k split steps with the correction held, then the fresh correction at the new state, the
hold error e = |c_held - c_fresh| / |du/dt| and a few regime statistics (energy per field, band
enstrophy) for the online envelope clause. Modes:

- "sync": the correction held over the interval, refreshed at its end (``split3d.advance``);
- "extrap": the correction extrapolated linearly from the last two refreshes across the interval
  (free: no extra network or solver work);
- "spec": a look-ahead of k steps with the correction held gives a predicted end state, the
  network there gives the end value, and the interval runs with the correction ramped from the
  start value to it (one extra solver pass per interval, no extra network call).

Holds of a given k compile once; the executor dispatches one call per interval.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from frost.cfd import mhd3d, spectral3d as s3, split3d
from frost.runtime.executor import Interval
from frost.runtime.reuse import ReuseClass, check_hold


def _rhs_norm(grid, u_hat, c_fresh, physics: str):
    if physics == "mhd":
        full = mhd3d.nonlinear(grid, u_hat) + mhd3d.forcing(grid, u_hat) + grid.linear * u_hat + c_fresh
    else:
        full = s3.nonlinear(grid, u_hat) + s3.forcing(grid, u_hat) + grid.linear * u_hat + c_fresh
    return jnp.sqrt(jnp.mean(s3.to_real(grid, full) ** 2, axis=(-4, -3, -2, -1)))


def _regime(grid, u_hat, physics: str):
    if physics == "mhd":
        ek, em = mhd3d.energies(grid, u_hat)
        return {"e_kin": jnp.mean(ek), "e_mag": jnp.mean(em), "mag_over_kin": jnp.mean(em) / jnp.mean(ek)}
    return {"e_kin": jnp.mean(s3.energy_of(grid, u_hat))}


def _make_hold(grid, model, dt: float):
    base, (physics, substeps) = split3d.base_of(model), split3d.physics_of(model)

    def steps(u, c0, c1, k):
        def body(j, x):
            c = c0 + ((j + 1.0) / k) * (c1 - c0)  # the value acting over step j (held when c1 = c0)
            return split3d.split_step(grid, x, c, dt, base, physics, substeps)
        return jax.lax.fori_loop(0, k, body, u)

    def close(u, c_used_end, k):
        c_new = split3d.correction(grid, model, u)
        num = jnp.sqrt(jnp.mean(s3.to_real(grid, c_used_end - c_new) ** 2, axis=(-4, -3, -2, -1)))
        e = num / _rhs_norm(grid, u, c_new, physics)
        finite = jnp.all(jnp.isfinite(u.real)) & jnp.all(jnp.isfinite(e))
        return c_new, jnp.mean(e), jnp.max(e), finite, _regime(grid, u, physics)

    @partial(jax.jit, static_argnames=("k", "mode"))
    def hold(u, c, c_prev, *, k: int, mode: str):
        if mode == "sync":
            u1 = steps(u, c, c, k)
            c_end = c
        elif mode == "extrap":
            c_end = c + (c - c_prev)
            u1 = steps(u, c, c_end, k)
        elif mode == "spec":
            u_hat = steps(u, c, c, k)  # look-ahead with the correction held
            c_end = split3d.correction(grid, model, u_hat)
            u1 = steps(u, c, c_end, k)
        else:
            raise ValueError(mode)
        c_new, e_mean, e_max, finite, regime = close(u1, c_end, k)
        if mode == "spec":
            c_new = c_end  # transported: the end value was computed at the predicted state
        return u1, c_new, e_mean, e_max, finite, regime

    return hold


class HoldBackend:
    reuse = ReuseClass.VALUE

    def __init__(self, grid, model, u_hat, dt: float = 0.04):
        check_hold(self.reuse, "output")
        self.grid, self.model, self.dt = grid, model, dt
        self.u = u_hat
        self.c = split3d.correction(grid, model, u_hat)
        self.c_prev = self.c
        self.step = 0
        self.network_calls = 1
        self._hold = _make_hold(grid, model, dt)

    def time(self) -> float:
        return float(self.step)

    def steps_left(self, horizon_steps: float) -> int:
        return max(0, int(horizon_steps) - self.step)

    def checkpoint(self):
        return (self.u, self.c, self.c_prev, self.step)  # jax arrays are immutable: no copy

    def restore(self, ck) -> None:
        self.u, self.c, self.c_prev, self.step = ck

    def finish(self) -> None:
        jax.block_until_ready(self.u)

    def interval(self, k: int, mode: str) -> Interval:
        u1, c_new, e_mean, e_max, finite, regime = self._hold(self.u, self.c, self.c_prev, k=k, mode=mode)
        self.c_prev, self.c, self.u = self.c, c_new, u1
        self.step += k
        self.network_calls += 1
        # one host read per interval (a handful of scalars), so the clauses can decide
        extra = {key: float(v) for key, v in {"hold_error": e_mean, "worst": e_max, **regime}.items()}
        return Interval(t=float(self.step), k=k, staleness=extra["hold_error"], finite=bool(finite), extra=extra)
