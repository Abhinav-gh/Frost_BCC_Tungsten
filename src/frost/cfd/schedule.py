"""Multirate schedules for a learned closure in workload W1 (mechanism M1).

The closure network runs once per outer step of k solver steps. Between refreshes
the runtime holds one of:

| mode | held | applied each stage |
|---|---|---|
| ``none`` | nothing (the coarse solver alone) | - |
| ``smag`` | nothing; Smagorinsky every stage | fresh |
| ``value`` | the tendency at the refresh | constant |
| ``operator`` | the stencil coefficients | to the fresh stage state |
| ``residual`` | (learned - Smagorinsky) tendency at the refresh | plus fresh Smagorinsky |
| ``value_adv`` | the tendency, carried with the flow each step | constant within a step |
| ``operator_adv`` | the coefficients, carried with the flow each step | to the fresh stage state |

At k = 1 ``operator`` is the ML solver's own every-step form, B0. Everything runs
inside ``lax.scan`` (the CFD analogue of a CUDA graph), so a timing measures the
schedule rather than Python dispatch.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from frost.cfd.closure import advect, apply_operator, coefficients, smagorinsky
from frost.cfd.spectral import Grid, advection_hat, step, to_hat, to_real

MODES = ("none", "smag", "value", "operator", "residual", "value_adv", "operator_adv")


def _refresh(grid: Grid, model, w_hat, mode: str, cs: float):
    """What the runtime caches at a refresh."""
    if mode in ("none", "smag"):
        return jnp.zeros((), w_hat.real.dtype)
    coef = coefficients(grid, model, w_hat)
    if mode in ("operator", "operator_adv"):
        return coef
    tend = apply_operator(grid, coef, w_hat)
    if mode == "residual":
        tend = tend - smagorinsky(grid, w_hat, cs)
    if mode == "value_adv":
        return to_real(grid, tend)[..., None]  # held in real space so it can be carried
    return tend


def _inner(grid: Grid, w_hat, held, mode: str, dt: float, cs: float):
    """One solver step with the held quantity; returns (state, held)."""
    if mode == "none":
        return step(grid, w_hat, dt), held
    if mode == "smag":
        return step(grid, w_hat, dt, lambda wh: smagorinsky(grid, wh, cs)), held
    if mode == "value":
        return step(grid, w_hat, dt, lambda wh: held), held
    if mode == "residual":
        return step(grid, w_hat, dt, lambda wh: smagorinsky(grid, wh, cs) + held), held
    if mode == "operator":
        return step(grid, w_hat, dt, lambda wh: apply_operator(grid, held, wh)), held
    if mode == "value_adv":
        tend = to_hat(grid, held[..., 0])
        new = step(grid, w_hat, dt, lambda wh: tend)
        return new, advect(grid, held, w_hat, dt)
    if mode == "operator_adv":
        new = step(grid, w_hat, dt, lambda wh: apply_operator(grid, held, wh))
        return new, advect(grid, held, w_hat, dt)
    raise ValueError(mode)


def outer_step(grid: Grid, model, w_hat, mode: str, k: int, dt: float, cs: float = 0.17):
    """Refresh once, then k solver steps."""
    held = _refresh(grid, model, w_hat, mode, cs)

    def body(carry, _):
        wh, hd = carry
        return _inner(grid, wh, hd, mode, dt, cs), None

    (w_hat, _), _ = jax.lax.scan(body, (w_hat, held), None, length=k)
    return w_hat


@partial(jax.jit, static_argnames=("grid", "mode", "k", "dt", "n_records", "record_every", "cs"))
def rollout(grid: Grid, model, w_hat, *, mode: str, k: int, dt: float, n_records: int,
            record_every: int, cs: float = 0.17):
    """States after every ``record_every`` steps (a multiple of k), ``n_records`` times."""
    if record_every % k:
        raise ValueError("record_every must be a multiple of k")

    def record(wh, _):
        def outer(w, _):
            return outer_step(grid, model, w, mode, k, dt, cs), None

        wh, _ = jax.lax.scan(outer, wh, None, length=record_every // k)
        return wh, wh

    _, states = jax.lax.scan(record, w_hat, None, length=n_records)
    return states  # (n_records, *batch, n, n//2+1)


@partial(jax.jit, static_argnames=("grid", "mode", "k", "dt", "n_outer", "cs"))
def advance(grid: Grid, model, w_hat, *, mode: str, k: int, dt: float, n_outer: int,
            cs: float = 0.17):
    """The timed kernel: n_outer outer steps (n_outer * k solver steps), final state only."""

    def outer(w, _):
        return outer_step(grid, model, w, mode, k, dt, cs), None

    w_hat, _ = jax.lax.scan(outer, w_hat, None, length=n_outer)
    return w_hat



# ------------------------------------------------------------------ certified intervals
def _certified_body(grid: Grid, model, k: int, dt: float, e_star: float):
    """Hold the learned operator k steps; if any member's hold error e = ||T(held) - T(fresh)|| /
    ||RHS|| at the interval's end exceeds e_star (or is non-finite), redo the interval as two
    halves. The refresh that measures e is the next interval's, so the check costs no extra
    network call (the spectral counterpart of ``frost.cfd.finite_volume``'s)."""

    def run_hold(c, w, n):
        return jax.lax.scan(lambda x, _: (step(grid, x, dt, lambda wh: apply_operator(grid, c, wh)), None),
                            w, None, length=n)[0]

    def hold_error(c_held, c_fresh, w):
        t_held = to_real(grid, apply_operator(grid, c_held, w))
        t_fresh = to_real(grid, apply_operator(grid, c_fresh, w))
        full = to_real(grid, (advection_hat(grid, w) + grid.forcing_hat + grid.linear * w
                              + apply_operator(grid, c_fresh, w)) * grid.mask)
        num = jnp.sqrt(jnp.mean((t_held - t_fresh) ** 2, axis=(-2, -1)))
        return num / jnp.sqrt(jnp.mean(full**2, axis=(-2, -1)))

    def body(carry, _):
        w, c, n_rb = carry
        wk = run_hold(c, w, k)
        ck = coefficients(grid, model, wk)
        e = hold_error(c, ck, wk)
        bad = ~jnp.isfinite(e) | (e > e_star) | ~jnp.all(jnp.isfinite(wk.real), axis=(-2, -1))

        def redo(_):
            h = k // 2
            wm = run_hold(c, w, h)
            wk2 = run_hold(coefficients(grid, model, wm), wm, k - h)
            ck2 = coefficients(grid, model, wk2)
            sel = bad[:, None, None]
            return jnp.where(sel, wk2, wk), jnp.where(sel[..., None], ck2, ck)

        wk, ck = jax.lax.cond(jnp.any(bad), redo, lambda _: (wk, ck), None)
        return (wk, ck, n_rb + jnp.any(bad).astype(jnp.int32)), jnp.mean(e)

    return body


@partial(jax.jit, static_argnames=("grid", "k", "dt", "e_star", "n_records", "record_every"))
def cert_rollout(grid: Grid, model, w_hat, *, k: int, dt: float, e_star: float, n_records: int,
                 record_every: int):
    body = _certified_body(grid, model, k, dt, e_star)

    def record(carry, _):
        carry, es = jax.lax.scan(body, carry, None, length=record_every // k)
        return carry, (carry[0], jnp.mean(es))

    init = (w_hat, coefficients(grid, model, w_hat), jnp.zeros((), jnp.int32))
    (_, _, n_rb), (states, es) = jax.lax.scan(record, init, None, length=n_records)
    return states, n_rb, jnp.mean(es)


@partial(jax.jit, static_argnames=("grid", "k", "dt", "e_star", "n_outer"))
def cert_advance(grid: Grid, model, w_hat, *, k: int, dt: float, e_star: float, n_outer: int):
    body = _certified_body(grid, model, k, dt, e_star)
    init = (w_hat, coefficients(grid, model, w_hat), jnp.zeros((), jnp.int32))
    (w_hat, _, _), _ = jax.lax.scan(body, init, None, length=n_outer)
    return w_hat
