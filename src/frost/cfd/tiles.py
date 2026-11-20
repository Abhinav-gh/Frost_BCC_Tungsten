"""Tile-sparse refresh of a learned closure (mechanism M4; tests 14-15).

The closure field is cut into t x t tiles. Each step a fixed budget of m tiles is
refreshed: those whose input changed most since their own last refresh, with any
tile older than ``age_max`` steps forced in. Every other tile keeps its held
coefficients (the operator hold of ``frost.cfd.schedule``). The fixed budget keeps
shapes static under XLA (SBNet's gather/scatter on a capacity bucket rather than a
dynamic mask), so the cost per step is the same every step:
- gathering m patches with a halo of one cell per CNN layer;
- running the CNN unpadded ("valid"), which returns each patch's centre;
- scattering the centres back.

With m = n_tiles / k the CNN does on average the work of the M1 schedule at k, so
M1 and M4 can be compared at equal refresh budget. M1 spends the budget uniformly
in time; M4 spends it where the input moved. On fronts M4 should win. On
homogeneous turbulence, where demand is near uniform (§13), it should not.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from frost.cfd import closure
from frost.cfd.spectral import Grid, step, to_real


def _tile_starts(n: int, t: int):
    nt = n // t
    ti, tj = jnp.meshgrid(jnp.arange(nt), jnp.arange(nt), indexing="ij")
    return (ti * t).reshape(-1), (tj * t).reshape(-1)


def tile_scores(a, b, t: int):
    """Mean |a - b| per tile: (B, n, n) -> (B, n_tiles), row-major tiles."""
    bsz, n, _ = a.shape
    nt = n // t
    d = jnp.abs(a - b).reshape(bsz, nt, t, nt, t).mean(axis=(2, 4))
    return d.reshape(bsz, nt * nt)


def gather(field, idx, t: int, halo: int):
    """field (B, n, n, C), idx (B, m) tile ids -> (B, m, t + 2h, t + 2h, C), periodic."""
    n = field.shape[1]
    padded = jnp.pad(field, ((0, 0), (halo, halo), (halo, halo), (0, 0)), mode="wrap")
    si, sj = _tile_starts(n, t)
    size = t + 2 * halo

    def one(f, i):
        return jax.lax.dynamic_slice(f, (si[i], sj[i], 0), (size, size, f.shape[-1]))

    return jax.vmap(lambda f, ids: jax.vmap(lambda i: one(f, i))(ids))(padded, idx)


def scatter(field, patches, idx, t: int):
    """Write patches (B, m, t, t, C) into field (B, n, n, C) at tiles idx (B, m)."""
    n = field.shape[1]
    si, sj = _tile_starts(n, t)
    r = si[idx][..., None, None] + jnp.arange(t)[:, None]   # (B, m, t, 1)
    c = sj[idx][..., None, None] + jnp.arange(t)[None, :]   # (B, m, 1, t)
    b = jnp.arange(field.shape[0])[:, None, None, None]
    r, c = jnp.broadcast_arrays(r, c)
    b = jnp.broadcast_to(b, r.shape)
    return field.at[b, r, c].set(patches)


def select(score, age, m: int, age_max: int):
    """The m tiles to refresh: forced if age >= age_max, then largest score."""
    priority = score + jnp.where(age >= age_max, 1e30, 0.0)
    return jax.lax.top_k(priority, m)[1]


def _refresh_tiles(grid: Grid, model, w_hat, raw, w_ref, age, m: int, t: int, age_max: int):
    w = to_real(grid, w_hat)
    idx = select(tile_scores(w, w_ref, t), age, m, age_max)
    halo = len(model["params"])  # one cell per 3x3 layer
    feats = closure.features(grid, w_hat, model["scale"])
    patches = gather(feats, idx, t, halo)
    bsz, mm = idx.shape
    flat = patches.reshape(bsz * mm, *patches.shape[2:])
    out = closure.apply_cnn(model["params"], flat, padding="valid")
    out = out.reshape(bsz, mm, t, t, out.shape[-1])
    raw = scatter(raw, out, idx, t)
    w_ref = scatter(w_ref[..., None], gather(w[..., None], idx, t, 0), idx, t)[..., 0]
    age = (age + 1).at[jnp.arange(bsz)[:, None], idx].set(0)
    return raw, w_ref, age


def _tiles_step(grid: Grid, model, carry, m: int, t: int, age_max: int, dt: float):
    w_hat, raw, w_ref, age = carry
    raw, w_ref, age = _refresh_tiles(grid, model, w_hat, raw, w_ref, age, m, t, age_max)
    coef = closure.coef_from_raw(grid, model, raw)  # cheap; the held operator, once per step
    w_hat = step(grid, w_hat, dt, lambda wh: closure.apply_operator(grid, coef, wh))
    return w_hat, raw, w_ref, age


def _init(grid: Grid, model, w_hat):
    """One dense refresh to start from (amortised over a run, as B0's first call is)."""
    raw = closure.apply_cnn(model["params"], closure.features(grid, w_hat, model["scale"]))
    return raw, to_real(grid, w_hat)


@partial(jax.jit, static_argnames=("grid", "m", "t", "age_max", "dt", "n_records", "record_every"))
def rollout(grid: Grid, model, w_hat, *, m: int, t: int, age_max: int, dt: float,
            n_records: int, record_every: int):
    coef, w_ref = _init(grid, model, w_hat)
    age = jnp.zeros((w_hat.shape[0], (grid.n // t) ** 2), jnp.int32)

    def record(carry, _):
        carry, _ = jax.lax.scan(lambda c, _: (_tiles_step(grid, model, c, m, t, age_max, dt), None),
                                carry, None, length=record_every)
        return carry, carry[0]

    _, states = jax.lax.scan(record, (w_hat, coef, w_ref, age), None, length=n_records)
    return states


@partial(jax.jit, static_argnames=("grid", "m", "t", "age_max", "dt", "n_steps"))
def advance(grid: Grid, model, w_hat, *, m: int, t: int, age_max: int, dt: float, n_steps: int):
    coef, w_ref = _init(grid, model, w_hat)
    age = jnp.zeros((w_hat.shape[0], (grid.n // t) ** 2), jnp.int32)
    carry, _ = jax.lax.scan(lambda c, _: (_tiles_step(grid, model, c, m, t, age_max, dt), None),
                            (w_hat, coef, w_ref, age), None, length=n_steps)
    return carry[0]
