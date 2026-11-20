"""The ML split operator on the spectral solver (Dresdner et al. 2023), and Frost's ways of
running it (workload W1, accuracy-first form).

    omega_hat(t + h) = PhysicsStep(omega_hat(t)) + h c_hat(t),
    c = s_out * EPD(s_in * (u, v)),   (u, v) from omega by the velocity solve.

- **PhysicsStep** is ``spectral.step`` on a grid with the exponential filter
  (``Grid(n, dealias="exp")``), the stronger classical setting.
- **The correction is explicit Euler outside the RK4 stages.** Dresdner et al. found
  this split form more accurate, more stable to train and 4x cheaper than a correction
  inside RK4; Frost's earlier closures sat inside the stages.
- **Inputs are velocity only.** In their comparison, models given vorticity (alone or
  with velocity) plateau near tau 4 where velocity-only models reach 6-7; Frost's
  earlier closures took (u, v, omega) and plateaued at 4.2-4.5.
- **The network** is an encoder-process-decoder CNN: a 5x5 encoder to 64 channels,
  residual blocks of dilated 3x3 convolutions with ReLUs, a 5x5 decoder, periodic
  padding. The decoder starts at zero, so an untrained model is the physics step.

The network costs several physics steps, which is why Dresdner et al. found no speedup
over spectral solvers. Their suggested remedy, applying the correction less often, is
the hold Frost certifies:

| mode | between refreshes |
|---|---|
| ``value`` | c from the refresh, applied every step (k = 1 is B0, the published model) |
| ``advected`` | c carried with the flow, semi-Lagrangian |
| certified | ``value``; an interval whose hold error exceeds e* is redone as two halves |

A hold may span the records a rollout emits: the step index, not the record, decides
the refresh.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from frost.cfd.closure import advect
from frost.cfd.spectral import Grid, advection_hat, step, to_hat, to_real, velocity


@dataclass(frozen=True)
class SplitStatic:
    """Network settings that select code: hashed into the treedef, never traced."""

    rates: tuple[int, ...] = (1, 2, 4, 2, 1)


jax.tree_util.register_static(SplitStatic)

MODES = ("value", "advected")


# ------------------------------------------------------------------ the network
def _conv_init(key, k: int, cin: int, cout: int, zero: bool = False) -> dict:
    shape = (k, k, cin, cout)
    w = (jnp.zeros(shape) if zero else
         (2.0 / (k * k * cin)) ** 0.5 * jax.random.truncated_normal(key, -2.0, 2.0, shape))
    return {"w": w.astype(jnp.float32), "b": jnp.zeros((cout,), jnp.float32)}


def init_epd(key, in_ch: int = 2, hidden: int = 64, blocks: int = 2, rates=(1, 2, 4, 2, 1),
             io_kernel: int = 5, kernel: int = 3, out_ch: int = 1) -> dict:
    keys = jax.random.split(key, 2 + blocks * len(rates))
    proc = [[_conv_init(keys[1 + b * len(rates) + i], kernel, hidden, hidden) for i in range(len(rates))]
            for b in range(blocks)]
    return {"enc": _conv_init(keys[0], io_kernel, in_ch, hidden), "proc": proc,
            "dec": _conv_init(keys[-1], io_kernel, hidden, out_ch, zero=True)}


def _conv(x, p: dict, dilation: int = 1):
    pad = dilation * (p["w"].shape[0] // 2)
    x = jnp.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="wrap")
    return jax.lax.conv_general_dilated(x, p["w"], (1, 1), "VALID", rhs_dilation=(dilation, dilation),
                                        dimension_numbers=("NHWC", "HWIO", "NHWC")) + p["b"]


def apply_epd(params: dict, x, rates: tuple[int, ...]):
    """(B, n, n, C_in) -> (B, n, n, C_out)."""
    h = _conv(x, params["enc"])
    for block in params["proc"]:
        r = h
        for p, d in zip(block, rates):
            r = _conv(jax.nn.relu(r), p, d)
        h = h + r
    return _conv(jax.nn.relu(h), params["dec"])


def epd_flops(params: dict, n: int) -> float:
    """Multiply-adds x 2 for one call on an n x n field."""
    convs = [params["enc"], params["dec"]] + [p for block in params["proc"] for p in block]
    return float(sum(2 * p["w"].size * n * n for p in convs))


def init_model(key, scale_in: float = 0.2, scale_out: float = 0.01, rates=(1, 2, 4, 2, 1), **kw) -> dict:
    return {"params": init_epd(key, rates=rates, **kw), "scale_in": scale_in, "scale_out": scale_out,
            "static": SplitStatic(tuple(rates))}


# ------------------------------------------------------------------ the split step
def correction(grid: Grid, model: dict, w_hat):
    """The learned tendency c_hat, zero mean, on the grid's kept band."""
    lead = w_hat.shape[:-2]
    u, v = velocity(grid, w_hat)
    x = (model["scale_in"] * jnp.stack([u, v], axis=-1)).reshape(-1, grid.n, grid.n, 2)
    c = model["scale_out"] * apply_epd(model["params"], x, model["static"].rates)[..., 0]
    return to_hat(grid, c.reshape(*lead, grid.n, grid.n)).at[..., 0, 0].set(0.0)


def split_step(grid: Grid, w_hat, c_hat, dt: float):
    return step(grid, w_hat, dt) + dt * c_hat


def _hold_body(grid: Grid, model: dict, mode: str, k: int, dt: float):
    """One split step under a hold of k steps; the carry (w_hat, held, step index)
    refreshes the correction whenever the index is a multiple of k, so a hold may span
    record boundaries."""

    def fresh(w):
        c = correction(grid, model, w)
        return c if mode == "value" else to_real(grid, c)[..., None]

    def body(carry, _):
        w, c, i = carry
        c = fresh(w) if k == 1 else jax.lax.cond(i % k == 0, lambda: fresh(w), lambda: c)
        if mode == "value":
            return (split_step(grid, w, c, dt), c, i + 1), None
        if mode == "advected":
            ch = to_hat(grid, c[..., 0]).at[..., 0, 0].set(0.0)
            return (split_step(grid, w, ch, dt), advect(grid, c, w, dt), i + 1), None
        raise ValueError(mode)

    return body


def _hold_init(grid: Grid, mode: str, w_hat):
    held = jnp.zeros_like(w_hat) if mode == "value" else jnp.zeros((*w_hat.shape[:-2], grid.n, grid.n, 1))
    return (w_hat, held, jnp.zeros((), jnp.int32))


@partial(jax.jit, static_argnames=("grid", "mode", "k", "dt", "n_records", "record_every"))
def rollout(grid: Grid, model: dict, w_hat, *, mode: str = "value", k: int = 1, dt: float, n_records: int,
            record_every: int):
    """States after every ``record_every`` steps, ``n_records`` times."""
    body = _hold_body(grid, model, mode, k, dt)

    def record(carry, _):
        carry = jax.lax.scan(body, carry, None, length=record_every)[0]
        return carry, carry[0]

    return jax.lax.scan(record, _hold_init(grid, mode, w_hat), None, length=n_records)[1]


@partial(jax.jit, static_argnames=("grid", "mode", "k", "dt", "n_steps"))
def advance(grid: Grid, model: dict, w_hat, *, mode: str = "value", k: int = 1, dt: float, n_steps: int):
    """The timed kernel: n_steps split steps, final state only."""
    return jax.lax.scan(_hold_body(grid, model, mode, k, dt), _hold_init(grid, mode, w_hat), None,
                        length=n_steps)[0][0]


# ------------------------------------------------------------------ certified intervals
def hold_error(grid: Grid, c_held, c_fresh, w_hat):
    """||c_held - c_fresh|| / ||full tendency at w_hat||, per member."""
    full = to_real(grid, (advection_hat(grid, w_hat) + grid.forcing_hat + grid.linear * w_hat + c_fresh)
                   * grid.mask)
    num = jnp.sqrt(jnp.mean(to_real(grid, c_held - c_fresh) ** 2, axis=(-2, -1)))
    return num / jnp.sqrt(jnp.mean(full**2, axis=(-2, -1)))


def _certified_body(grid: Grid, model: dict, k: int, dt: float, e_star: float):
    """Hold c for k steps; the next interval's refresh measures the hold error at no extra
    network call. A member over e_star (or non-finite) has its interval redone as two
    halves with a refresh between (MD's rollback, in CFD). Emits the k states."""

    def run_hold(c, w, n):
        return jax.lax.scan(lambda x, _: (split_step(grid, x, c, dt),) * 2, w, None, length=n)

    def body(carry, _):
        w, c, n_rb = carry
        wk, traj = run_hold(c, w, k)
        ck = correction(grid, model, wk)
        e = hold_error(grid, c, ck, wk)
        bad = ~jnp.isfinite(e) | (e > e_star) | ~jnp.all(jnp.isfinite(wk.real), axis=(-2, -1))

        def redo(_):
            h = k // 2
            wm, t1 = run_hold(c, w, h)
            wk2, t2 = run_hold(correction(grid, model, wm), wm, k - h)
            ck2 = correction(grid, model, wk2)
            sel = bad[..., None, None]
            return (jnp.where(sel, wk2, wk), jnp.where(sel, ck2, ck),
                    jnp.where(sel[None], jnp.concatenate([t1, t2]), traj))

        wk, ck, traj = jax.lax.cond(jnp.any(bad), redo, lambda _: (wk, ck, traj), None)
        return (wk, ck, n_rb + jnp.any(bad).astype(jnp.int32)), (traj, jnp.mean(e))

    return body


@partial(jax.jit, static_argnames=("grid", "k", "dt", "e_star", "n_records", "record_every"))
def cert_rollout(grid: Grid, model: dict, w_hat, *, k: int, dt: float, e_star: float, n_records: int,
                 record_every: int):
    """States after every ``record_every`` steps; intervals of k steps may span records."""
    total = n_records * record_every
    body = _certified_body(grid, model, k, dt, e_star)
    init = (w_hat, correction(grid, model, w_hat), jnp.zeros((), jnp.int32))
    (_, _, n_rb), (traj, es) = jax.lax.scan(body, init, None, length=-(-total // k))
    flat = traj.reshape(-1, *traj.shape[2:])[:total]
    return flat[record_every - 1 :: record_every], n_rb, jnp.mean(es)


@partial(jax.jit, static_argnames=("grid", "k", "dt", "e_star", "n_steps"))
def cert_advance(grid: Grid, model: dict, w_hat, *, k: int, dt: float, e_star: float, n_steps: int):
    body = _certified_body(grid, model, k, dt, e_star)
    init = (w_hat, correction(grid, model, w_hat), jnp.zeros((), jnp.int32))
    return jax.lax.scan(lambda c, x: (body(c, x)[0], None), init, None, length=n_steps // k)[0][0]
