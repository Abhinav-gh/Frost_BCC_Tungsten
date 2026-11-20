"""NeuralGCM backend for the runtime: the learned physics refreshed every k physics intervals.

The operator is NeuralGCM's learned physics (a per-column network), value-consumed: its tendency is
added to the dynamical core's explicit terms and held over the core's substeps
(``frost.gcm.neuralgcm_split.SplitModel``: ``physics`` and ``finish`` composed from public pieces).
One runtime interval is k physics intervals:

- "sync": the physics at the interval's start, its tendency frozen for all k (the released model
  holds it for one interval, i.e. k = 1);
- "relax": the physics at the start; on the following k - 1 intervals the tendency is the fitted
  relaxation T_n + J (x - x_n), J the per-field, per-level least-squares slope of the tendency
  against the state between the last two refreshes, clipped to damping
  (``experiments/e11_gcm/forecast_skill.py``'s ``relaxK`` token, the same code path).

The interval's record carries the dry-mass statistic (global mean surface-pressure log, which the
dynamics conserves and a drifting physics hold would not) for the online regime clause. The skill
certificate itself is the paired forecast test (``forecast_skill.py``), run offline over many starts.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from frost.runtime.executor import Interval
from frost.runtime.reuse import ReuseClass, check_hold


def tendency_like(tend, state):
    import dataclasses

    return dataclasses.replace(tend, **{f.name: (None if getattr(tend, f.name) is None else getattr(state, f.name))
                                        for f in dataclasses.fields(tend)})


def relaxation_slope(t_new, t_old, s_new, s_old):
    def slope(tn, to, sn, so):
        ds, dt = sn - so, tn - to
        axes = tuple(range(1, ds.ndim))
        num = jnp.sum(dt * ds, axis=axes, keepdims=True)
        den = jnp.sum(ds * ds, axis=axes, keepdims=True)
        return jnp.minimum(jnp.where(den > 0, num / jnp.maximum(den, 1e-30), 0.0), 0.0)

    return jax.tree_util.tree_map(slope, t_new, t_old, s_new, s_old)


class GCMBackend:
    reuse = ReuseClass.VALUE

    def __init__(self, model, params, x0, forcings, chunk: int = 1):
        """``chunk``: refresh blocks per compiled call. One runtime interval is then ``chunk`` blocks
        of k physics intervals each, so the host dispatches and synchronises once per chunk: one call
        per model hour cost the released schedule 14% (32.22 against 28.29 ms per model hour)."""
        check_hold(self.reuse, "output")
        self.chunk = int(chunk)
        self.m, self.params, self.forcings = model, params, forcings
        self.x = x0
        f0 = model.forcing_at(params, forcings, x0)
        self.t_last = model.physics(params, x0, f0)
        self.s_last = tendency_like(self.t_last, x0.state)
        self.slope = jax.tree_util.tree_map(lambda t: jnp.zeros(t.shape[:1] + (1,) * (t.ndim - 1), t.dtype), self.t_last)
        self.step = 0
        self.physics_calls = 1
        self.mass0 = float(self._mass(x0))
        self._run = {}

    @staticmethod
    def _mass(x):
        lsp = x.state.log_surface_pressure
        return jnp.real(lsp[..., 0, 0]).mean()  # the l = m = 0 coefficient: the global mean

    def time(self) -> float:
        return float(self.step)

    def steps_left(self, horizon: float) -> int:
        return max(0, int(horizon) - self.step)

    def checkpoint(self):
        return (self.x, self.t_last, self.s_last, self.slope, self.step)

    def restore(self, ck) -> None:
        self.x, self.t_last, self.s_last, self.slope, self.step = ck

    def finish(self) -> None:
        jax.block_until_ready(self.x)

    def _compiled(self, k: int, mode: str):
        key = (k, mode)
        if key not in self._run:
            m, params, forcings = self.m, self.params, self.forcings

            def block(x, t_last, s_last, slope):
                f = m.forcing_at(params, forcings, x)
                t_new = m.physics(params, x, f)
                s_now = tendency_like(t_new, x.state)
                slope_new = relaxation_slope(t_new, t_last, s_now, s_last) if mode == "relax" else slope

                def one(carry, j):
                    x, = carry
                    fj = m.forcing_at(params, forcings, x)
                    if mode == "relax":
                        sj = tendency_like(t_new, x.state)
                        held = jax.tree_util.tree_map(lambda t, a, sn, so: t + a * (sn - so), t_new, slope_new, sj, s_now)
                    else:
                        held = t_new
                    return (m.finish(params, x, held, fj),), None

                (x1,), _ = jax.lax.scan(one, (x,), jnp.arange(k))
                return x1, t_new, s_now, slope_new

            def run(x, t_last, s_last, slope):
                def body(carry, _):
                    return block(*carry), None

                return jax.lax.scan(body, (x, t_last, s_last, slope), None, length=self.chunk)[0]

            self._run[key] = jax.jit(run)
        return self._run[key]

    def interval(self, k: int, mode: str) -> Interval:
        """``k``: physics intervals in this runtime interval, i.e. ``chunk`` blocks of k // chunk."""
        if k % self.chunk:
            raise ValueError(f"interval {k} is not a multiple of the chunk {self.chunk}")
        hold = k // self.chunk
        x1, self.t_last, self.s_last, self.slope = self._compiled(hold, mode)(self.x, self.t_last, self.s_last, self.slope)
        self.x = x1
        self.step += k
        self.physics_calls += self.chunk
        mass = float(self._mass(x1))
        finite = bool(jnp.isfinite(mass))
        return Interval(t=float(self.step), k=k, staleness=0.0, finite=finite,
                        extra={"mass_drift": (mass - self.mass0), "mass": mass})
