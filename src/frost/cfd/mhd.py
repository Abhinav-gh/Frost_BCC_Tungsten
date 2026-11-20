"""Workload W2: 2D ideal MHD on astronomix with a learned corrector (tests 15 and later).

The solver is astronomix (finite volume, HLL, its own Pallas kernels on GPU), driven
only through its public ``time_integration`` with a fixed step. That call runs
inside our ``lax.scan``. The corrector follows astronomix's own ``CorrectorCNN``
design (Storcks & Buck 2025), written here as our own network:
- three 3x3 convolutions map the normalised primitive state (rho, vx, vy, p, Bx, By, Bz)
  to seven channels;
- the last three channels are an electric field E, and the magnetic correction is
  curl E, so div B is untouched;
- the correction is a tendency, applied after each solver step as
  ``state += correction * dt``, with astronomix's pressure floor.

That is the order astronomix's iteration-level update uses, done by operator
splitting in our loop so that the runtime controls when the network runs.

Schedules:
- **B0**: the corrector every step.
- **M1**: the raw network output refreshed every k steps and held. The curl is
  recomputed every step from the held E; it is cheap.
- **M4**: a fixed budget of m tiles refreshed every step: those whose input moved
  most since their own refresh, plus any older than ``age_max``
  (``frost.cfd.tiles``).
"""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from frost.cfd import closure, tiles

GAMMA = 5.0 / 3.0
N_VARS = 7


# ------------------------------------------------------------------ problems
def blast_wave(n: int, center=(0.0, 0.0), p_in: float = 10.0, radius: float = 0.1):
    """The Gardiner-Stone MHD blast wave on the periodic unit box."""
    x = (np.arange(n) + 0.5) / n - 0.5
    xx, yy = np.meshgrid(x, x, indexing="ij")
    dx_, dy_ = xx - center[0], yy - center[1]
    dx_ -= np.round(dx_)
    dy_ -= np.round(dy_)
    r = np.sqrt(dx_**2 + dy_**2)
    b = 1.0 / math.sqrt(2.0)
    return {"density": np.ones((n, n)), "velocity_x": np.zeros((n, n)),
            "velocity_y": np.zeros((n, n)), "gas_pressure": np.where(r < radius, p_in, 0.1),
            "magnetic_field_x": np.full((n, n), b), "magnetic_field_y": np.full((n, n), b),
            "magnetic_field_z": np.zeros((n, n))}


def stable_dt(fields: dict, n: int, cfl: float = 0.4) -> float:
    rho, p = fields["density"], fields["gas_pressure"]
    b2 = fields["magnetic_field_x"] ** 2 + fields["magnetic_field_y"] ** 2
    cf = np.sqrt(GAMMA * p / rho + b2 / rho).max()
    return cfl * (1.0 / n) / (cf + 1.0)


class Solver:
    """One fixed-dt astronomix step, traceable inside ``lax.scan``."""

    def __init__(self, fields: dict, n: int, dt: float, differentiable: bool = False):
        """``differentiable``: astronomix's public BACKWARDS mode, for training through the
        solver. Its default loop is a while_loop, which reverse mode cannot differentiate."""
        from astronomix.initial_condition_generation.construct_primitive_state import (
            construct_primitive_state,
        )
        from astronomix.option_classes.simulation_config import (
            BACKWARDS,
            FINITE_VOLUME,
            FORWARDS,
            HLL,
            PERIODIC_BOUNDARY,
            BoundarySettings,
            BoundarySettings1D,
            SimulationConfig,
            finalize_config,
        )
        from astronomix.option_classes.simulation_params import SimulationParams
        from astronomix.time_stepping import time_integration
        from astronomix.variable_registry.registered_variables import get_registered_variables

        periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
        config = SimulationConfig(solver_mode=FINITE_VOLUME, dimensionality=2, mhd=True,
                                  box_size=1.0, num_cells=n, riemann_solver=HLL,
                                  boundary_settings=BoundarySettings(x=periodic, y=periodic),
                                  fixed_timestep=True, num_timesteps=1,
                                  differentiation_mode=BACKWARDS if differentiable else FORWARDS)
        self.rv = get_registered_variables(config)
        state = construct_primitive_state(
            config=config, registered_variables=self.rv,
            **{k: jnp.asarray(v, dtype=jnp.float32) for k, v in fields.items()})
        self.config = finalize_config(config, state.shape)
        self.params = SimulationParams(t_end=dt, C_cfl=0.4, gamma=GAMMA)
        self._ti = time_integration
        self.state0 = state
        self.n, self.dt, self.dx = n, dt, 1.0 / n

    def step(self, state):
        return self._ti(state, self.config, self.params, self.rv)


def downsample(state, factor: int):
    """Block average of the primitive state, (7, nf, nf) -> (7, nf/f, nf/f)."""
    c, n, _ = state.shape
    return state.reshape(c, n // factor, factor, n // factor, factor).mean(axis=(2, 4))


# ------------------------------------------------------------------ the corrector
def curl_e(e, dx: float):
    """Curl of (Ex, Ey, Ez) in 2D, periodic central differences: div of the result is 0."""
    def ddx(f):
        return (jnp.roll(f, -1, axis=-2) - jnp.roll(f, 1, axis=-2)) / (2 * dx)

    def ddy(f):
        return (jnp.roll(f, -1, axis=-1) - jnp.roll(f, 1, axis=-1)) / (2 * dx)

    ex, ey, ez = e[..., 0, :, :], e[..., 1, :, :], e[..., 2, :, :]
    return jnp.stack([ddy(ez), -ddx(ez), ddx(ey) - ddy(ex)], axis=-3)


def normalise(state, scale):
    """(..., 7, n, n) -> (..., n, n, 7) normalised network input."""
    mean = jnp.asarray(scale["mean"])[:, None, None]
    std = jnp.asarray(scale["std"])[:, None, None]
    return jnp.moveaxis((state - mean) / std, -3, -1)


def raw_output(model, state):
    """The expensive part: the CNN on the whole grid. (B, 7, n, n) -> (B, n, n, 7)."""
    return closure.apply_cnn(model["params"], normalise(state, model["scale"]))


def correction(model, raw, dx: float):
    """The cheap part: (B, n, n, 7) network output -> (B, 7, n, n) tendency, B via curl E.

    The output is soft-bounded at three target standard deviations (3 tanh(o / 3)), so
    a network far outside its training distribution cannot inject an unbounded tendency.
    """
    out = jnp.moveaxis(3.0 * jnp.tanh(raw / 3.0), -1, -3) * jnp.asarray(model["scale"]["out"])[:, None, None]
    e = out[..., 4:, :, :]
    if model.get("static", closure.Static()).inplane:
        # 2D in-plane field: only E_z acts (curl gives Bx, By); Bz is never touched
        e = e.at[..., :2, :, :].set(0.0)
    return jnp.concatenate([out[..., :4, :, :], curl_e(e, dx)], axis=-3)


POSITIVE = (0, 3)  # density and pressure: corrected multiplicatively


def apply_correction(state, corr, dt: float):
    """Velocity and B additively; density and pressure as relative rates.

    rho <- rho exp(dt c_rho) and p <- p exp(dt c_p) cannot change sign. An additive
    correction near the shock drove density negative and NaN'd the HLL flux in both
    training attempts (test 15, 2026-10-01). Pressure keeps astronomix's floor as well.
    """
    add = state + corr * dt
    rel = state * jnp.exp(corr * dt)
    pos = jnp.zeros((state.shape[-3], 1, 1), bool).at[jnp.asarray(POSITIVE)].set(True)
    state = jnp.where(pos, rel, add)
    return state.at[..., 3, :, :].set(jnp.maximum(state[..., 3, :, :], 1e-4))


def init_corrector(key, hidden: int = 64, layers: int = 3):
    return closure.init_cnn(key, in_ch=N_VARS, hidden=hidden, layers=layers, out_ch=N_VARS)


# ------------------------------------------------------------------ schedules
def _solver_steps(solver: Solver, state, n: int):
    return jax.vmap(lambda s: jax.lax.scan(lambda c, _: (solver.step(c), None), s, None,
                                           length=n)[0])(state)


def run(solver: Solver, model, state, *, mode: str, n_steps: int, k: int = 1, m: int = 0,
        t: int = 32, age_max: int = 8, record_every: int | None = None):
    """Batched (B, 7, n, n) run. ``mode``: none, dense (B0 at k = 1, M1 at k > 1), tiles (M4)."""
    dt, dx = solver.dt, solver.dx
    step1 = jax.vmap(solver.step)

    def body(carry, i):
        s, raw, ref, age = carry
        s = step1(s)
        if mode == "none":
            return (s, raw, ref, age), None
        if mode == "dense":
            raw = jax.lax.cond(i % k == 0, lambda: raw_output(model, s), lambda: raw)
        elif mode == "tiles":
            x = normalise(s, model["scale"])
            # per tile: mean change of the normalised input since the tile's own refresh
            moved = jnp.linalg.norm(x - ref, axis=-1)
            score = tiles.tile_scores(moved, jnp.zeros_like(moved), t)
            idx = tiles.select(score, age, m, age_max)
            halo = len(model["params"])
            patches = tiles.gather(x, idx, t, halo)
            bsz, mm = idx.shape
            out = closure.apply_cnn(model["params"], patches.reshape(bsz * mm, *patches.shape[2:]),
                                    padding="valid").reshape(bsz, mm, t, t, N_VARS)
            raw = tiles.scatter(raw, out, idx, t)
            ref = tiles.scatter(ref, tiles.gather(x, idx, t, 0), idx, t)
            age = (age + 1).at[jnp.arange(bsz)[:, None], idx].set(0)
        s = apply_correction(s, correction(model, raw, dx), dt)
        return (s, raw, ref, age), (s if record_every else None)

    bsz = state.shape[0]
    raw0 = (raw_output(model, state) if mode != "none"
            else jnp.zeros((bsz, solver.n, solver.n, N_VARS)))
    ref0 = (normalise(state, model["scale"]) if mode == "tiles"
            else jnp.zeros((bsz, solver.n, solver.n, N_VARS)))
    age0 = jnp.zeros((bsz, (solver.n // t) ** 2), jnp.int32)
    (s, *_), states = jax.lax.scan(body, (state, raw0, ref0, age0), jnp.arange(n_steps))
    if record_every:
        return s, states[record_every - 1 :: record_every]
    return s, None


@partial(jax.jit, static_argnames=("solver", "mode", "n_steps", "k", "m", "t", "age_max",
                                   "record_every"))
def run_jit(solver, model, state, *, mode, n_steps, k=1, m=0, t=32, age_max=8,
            record_every=None):
    return run(solver, model, state, mode=mode, n_steps=n_steps, k=k, m=m, t=t, age_max=age_max,
               record_every=record_every)
