"""The ML split operator in 3D (workload W3): ``frost.cfd.split`` on ``frost.cfd.spectral3d``.

    u_hat(t + h) = PhysicsStep(u_hat(t)) + h c_hat(t),   c = P[s_out * EPD3(s_in * u)],

with a 3D encoder-process-decoder network (5^3 encoder and decoder, residual blocks of
dilated 3^3 convolutions, periodic padding; the decoder starts at zero, so an untrained
model is the physics step). The correction is projected onto divergence-free fields and
has zero mean. Holds and certified intervals are those of the 2D module: the step index,
not the record, decides a refresh, and a certified interval whose hold error exceeds e*
is redone as two halves.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from frost.cfd import mhd3d, spectral3d as s3
from frost.cfd.spectral3d import Grid3


@dataclass(frozen=True)
class Split3Static:
    rates: tuple[int, ...] = (1, 2, 4, 2, 1)
    arch: str = "epd"  # or "dw": depthwise-separable processor (dilated 3^3 depthwise, then 1x1x1)
    base: str = "none"  # the classical closure under the correction: none | cl | smag (s3.les_tendency)
    base_c: float = 0.0
    physics: str = "hydro"  # or "mhd": frost.cfd.mhd3d, a six-component state (u, b)
    substeps: int = 1  # physics steps per split step (MHD's Alfven speed halves the stable step)
    net_dtype: str = "fp32"  # or "bf16": the network's parameters and activations in bfloat16 (1.57x on A6000)
    stochastic: bool = False  # a second decoder head: the local amplitude of a noise term drawn at every refresh


jax.tree_util.register_static(Split3Static)

DIMS = ("NDHWC", "DHWIO", "NDHWC")


def _conv_init(key, k: int, cin: int, cout: int, zero: bool = False) -> dict:
    shape = (k, k, k, cin, cout)
    w = (jnp.zeros(shape) if zero else
         (2.0 / (k**3 * cin)) ** 0.5 * jax.random.truncated_normal(key, -2.0, 2.0, shape))
    return {"w": w.astype(jnp.float32), "b": jnp.zeros((cout,), jnp.float32)}


def init_epd3(key, in_ch: int = 3, hidden: int = 32, blocks: int = 2, rates=(1, 2, 4, 2, 1),
              io_kernel: int = 5, kernel: int = 3, out_ch: int = 3) -> dict:
    keys = jax.random.split(key, 2 + blocks * len(rates))
    proc = [[_conv_init(keys[1 + b * len(rates) + i], kernel, hidden, hidden) for i in range(len(rates))]
            for b in range(blocks)]
    return {"enc": _conv_init(keys[0], io_kernel, in_ch, hidden), "proc": proc,
            "dec": _conv_init(keys[-1], io_kernel, hidden, out_ch, zero=True)}


def _conv(x, p: dict, dilation: int = 1):
    pad = dilation * (p["w"].shape[0] // 2)
    x = jnp.pad(x, ((0, 0), (pad, pad), (pad, pad), (pad, pad), (0, 0)), mode="wrap")
    return jax.lax.conv_general_dilated(x, p["w"], (1, 1, 1), "VALID", rhs_dilation=(dilation,) * 3,
                                        dimension_numbers=DIMS) + p["b"]


def init_dw3(key, in_ch: int = 3, hidden: int = 32, blocks: int = 2, rates=(1, 2, 4, 2, 1),
             io_kernel: int = 3, kernel: int = 3, out_ch: int = 3) -> dict:
    """The EPD with a depthwise-separable processor: each layer is a dilated k^3 depthwise
    convolution followed by a 1x1x1 channel mix, ~(k^3 + C) / (k^3 C) of the dense cost."""
    keys = jax.random.split(key, 2 + 2 * blocks * len(rates))
    shape_dw = (kernel, kernel, kernel, 1, hidden)

    def layer(i):
        kd, kp = keys[1 + 2 * i], keys[2 + 2 * i]
        return {"dw": (2.0 / kernel**3) ** 0.5 * jax.random.truncated_normal(kd, -2.0, 2.0, shape_dw),
                "pw": (2.0 / hidden) ** 0.5 * jax.random.truncated_normal(kp, -2.0, 2.0, (hidden, hidden)),
                "b": jnp.zeros((hidden,), jnp.float32)}

    proc = [[layer(b * len(rates) + i) for i in range(len(rates))] for b in range(blocks)]
    return {"enc": _conv_init(keys[0], io_kernel, in_ch, hidden), "proc": proc,
            "dec": _conv_init(keys[-1], io_kernel, hidden, out_ch, zero=True)}


def _dw_layer(x, p: dict, dilation: int):
    k = p["dw"].shape[0]
    pad = dilation * (k // 2)
    x = jnp.pad(x, ((0, 0), (pad, pad), (pad, pad), (pad, pad), (0, 0)), mode="wrap")
    x = jax.lax.conv_general_dilated(x, p["dw"].astype(x.dtype), (1, 1, 1), "VALID", rhs_dilation=(dilation,) * 3,
                                     dimension_numbers=DIMS, feature_group_count=x.shape[-1])
    return x @ p["pw"] + p["b"]


def features_epd3(params: dict, x, rates: tuple[int, ...], arch: str = "epd"):
    """The encoder and processor: (B, n, n, n, C_in) -> the decoder's input, (B, n, n, n, hidden)."""
    h = _conv(x, params["enc"])
    for block in params["proc"]:
        r = h
        for p, d in zip(block, rates):
            r = _conv(jax.nn.relu(r), p, d) if arch == "epd" else _dw_layer(jax.nn.relu(r), p, d)
        h = h + r
    return jax.nn.relu(h)


def apply_epd3(params: dict, x, rates: tuple[int, ...], arch: str = "epd"):
    """(B, n, n, n, C_in) -> (B, n, n, n, C_out)."""
    return _conv(features_epd3(params, x, rates, arch), params["dec"])


def epd3_flops(params: dict, n: int) -> float:
    """Multiply-adds x 2 per call, per trajectory."""
    total = sum(2 * p["w"].size for p in (params["enc"], params["dec"]))
    for block in params["proc"]:
        for p in block:
            total += 2 * (p["w"].size if "w" in p else p["dw"].size + p["pw"].size)
    return float(total * n**3)


def init_model(key, scale_in: float = 1.0, scale_out: float = 0.01, rates=(1, 2, 4, 2, 1), arch: str = "epd",
               base: str = "none", base_c: float = 0.0, physics: str = "hydro", substeps: int = 1,
               stochastic: bool = False, **kw) -> dict:
    init = init_epd3 if arch == "epd" else init_dw3
    channels = 6 if physics == "mhd" else 3
    kw = {"in_ch": channels, "out_ch": channels, **kw}
    model = {"params": init(key, rates=rates, **kw), "scale_in": scale_in, "scale_out": scale_out,
             "static": Split3Static(tuple(rates), arch, base, float(base_c), physics, int(substeps))}
    return add_noise_head(model, jax.random.fold_in(key, 7)) if stochastic else model


def add_noise_head(model: dict, key, sigma_init: float = -4.0) -> dict:
    """A stochastic closure from a deterministic one (Guillaumin and Zanna 2021; Perezhogin et al. 2023): a second
    decoder head gives the local amplitude softplus(s) of a white noise term drawn at every refresh, so the correction
    is mean + amplitude x noise before the divergence-free projection. The head's weights start at zero and its bias at
    ``sigma_init`` (softplus(-4) = 0.018 of the mean head's unit), so a warm start begins as the deterministic model."""
    import dataclasses

    dec = model["params"]["dec"]
    k, hidden, out_ch = dec["w"].shape[0], dec["w"].shape[3], dec["w"].shape[4]
    head = _conv_init(key, k, hidden, out_ch, zero=True)
    head["b"] = jnp.full((out_ch,), sigma_init, jnp.float32)
    return {**model, "params": {**model["params"], "dec_sigma": head}, "noise_key": jax.random.PRNGKey(0),
            "static": dataclasses.replace(model["static"], stochastic=True)}


def _project(grid: Grid3, f_hat):
    return s3.project(grid, f_hat) if f_hat.shape[-4] == 3 else mhd3d.project(grid, f_hat)


def correction(grid: Grid3, model: dict, u_hat, step=None):
    """The learned tendency: divergence-free (each field), zero mean, on the grid's kept band.

    A stochastic model (``add_noise_head``) adds amplitude x noise when given ``step``, the index of the refresh: the
    noise is drawn from ``fold_in(model["noise_key"], step)``, independently at every point, field and trajectory, and
    a hold of k keeps that draw for k steps. Without ``step`` (a hold-error test, a deterministic caller) it is the mean.
    """
    lead = u_hat.shape[:-4]
    n, ch = grid.n, u_hat.shape[-4]
    u = s3.to_real(grid, u_hat)                                   # (..., ch, n, n, n)
    x = jnp.moveaxis(u, -4, -1).reshape(-1, n, n, n, ch) * model["scale_in"]
    params = model["params"]
    if model["static"].net_dtype == "bf16":
        x = x.astype(jnp.bfloat16)
        params = jax.tree_util.tree_map(lambda a: a.astype(jnp.bfloat16), params)
    st = model["static"]
    if st.stochastic and step is not None:
        h = features_epd3(params, x, st.rates, st.arch)
        c = _conv(h, params["dec"]).astype(jnp.float32)
        amp = jax.nn.softplus(_conv(h, params["dec_sigma"]).astype(jnp.float32))
        c = c + amp * jax.random.normal(jax.random.fold_in(model["noise_key"], step), c.shape, jnp.float32)
    else:
        c = apply_epd3(params, x, st.rates, st.arch).astype(jnp.float32)
    c = model["scale_out"] * c
    c = jnp.moveaxis(c.reshape(*lead, n, n, n, ch), -1, -4)
    return _project(grid, s3.to_hat(grid, c)).at[..., 0, 0, 0].set(0.0)


def physics_step(grid: Grid3, u_hat, dt: float, base: tuple[str, float] = ("none", 0.0), physics: str = "hydro",
                 substeps: int = 1):
    """``substeps`` steps of the physics (with the classical closure ``base``) over dt."""
    mod = mhd3d if physics == "mhd" else s3
    tend = mod.les_tendency(grid, *base)
    for _ in range(substeps):
        u_hat = mod.step(grid, u_hat, dt / substeps, tend)
    return u_hat


def split_step(grid: Grid3, u_hat, c_hat, dt: float, base: tuple[str, float] = ("none", 0.0),
               physics: str = "hydro", substeps: int = 1):
    """PhysicsStep (with the classical closure ``base``, if any) plus the held correction."""
    return physics_step(grid, u_hat, dt, base, physics, substeps) + dt * c_hat


def base_of(model: dict) -> tuple[str, float]:
    st = model["static"]
    return st.base, st.base_c


def physics_of(model: dict) -> tuple[str, int]:
    st = model["static"]
    return st.physics, st.substeps


def _hold_body(grid: Grid3, model: dict, k: int, dt: float, mode: str = "sync"):
    """One split step under a hold of k. ``mode``: "sync" holds the refreshed correction for k steps;
    "extrap" extrapolates it linearly from the last two refreshes across the hold (no extra cost)."""
    def body(carry, _):
        u, c, c_prev, i = carry
        if k == 1:
            c_now = correction(grid, model, u, step=i)
            return (split_step(grid, u, c_now, dt, base_of(model), *physics_of(model)), c_now, c_now, i + 1), None
        c, c_prev = jax.lax.cond(i % k == 0, lambda: (correction(grid, model, u, step=i), c), lambda: (c, c_prev))
        if mode == "extrap":
            frac = ((i % k) + 1.0) / k
            # the first hold has no history: c_prev is zero there, so extrapolate only from the second on
            c_use = jnp.where(i >= k, c + frac * (c - c_prev), c)
        else:
            c_use = c
        return (split_step(grid, u, c_use, dt, base_of(model), *physics_of(model)), c, c_prev, i + 1), None

    return body


@partial(jax.jit, static_argnames=("grid", "k", "dt", "n_records", "record_every", "mode"))
def rollout(grid: Grid3, model: dict, u_hat, *, k: int = 1, dt: float, n_records: int, record_every: int,
            mode: str = "sync"):
    body = _hold_body(grid, model, k, dt, mode)

    def record(carry, _):
        carry = jax.lax.scan(body, carry, None, length=record_every)[0]
        return carry, carry[0]

    init = (u_hat, jnp.zeros_like(u_hat), jnp.zeros_like(u_hat), jnp.zeros((), jnp.int32))
    return jax.lax.scan(record, init, None, length=n_records)[1]


@partial(jax.jit, static_argnames=("grid", "k", "dt", "n_steps", "mode"))
def advance(grid: Grid3, model: dict, u_hat, *, k: int = 1, dt: float, n_steps: int, mode: str = "sync"):
    init = (u_hat, jnp.zeros_like(u_hat), jnp.zeros_like(u_hat), jnp.zeros((), jnp.int32))
    return jax.lax.scan(_hold_body(grid, model, k, dt, mode), init, None, length=n_steps)[0][0]


def hold_error(grid: Grid3, c_held, c_fresh, u_hat):
    full = s3.nonlinear(grid, u_hat) + s3.forcing(grid, u_hat) + grid.linear * u_hat + c_fresh
    num = jnp.sqrt(jnp.mean(s3.to_real(grid, c_held - c_fresh) ** 2, axis=(-4, -3, -2, -1)))
    return num / jnp.sqrt(jnp.mean(s3.to_real(grid, full) ** 2, axis=(-4, -3, -2, -1)))


def _certified_body(grid: Grid3, model: dict, k: int, dt: float, e_star: float):
    def run_hold(c, u, n):
        return jax.lax.scan(lambda x, _: (split_step(grid, x, c, dt, base_of(model), *physics_of(model)),) * 2, u, None, length=n)

    def body(carry, _):
        u, c, n_rb = carry
        uk, traj = run_hold(c, u, k)
        ck = correction(grid, model, uk)
        e = hold_error(grid, c, ck, uk)
        bad = ~jnp.isfinite(e) | (e > e_star) | ~jnp.all(jnp.isfinite(uk.real), axis=(-4, -3, -2, -1))

        def redo(_):
            h = k // 2
            um, t1 = run_hold(c, u, h)
            uk2, t2 = run_hold(correction(grid, model, um), um, k - h)
            ck2 = correction(grid, model, uk2)
            sel = bad[..., None, None, None, None]
            return (jnp.where(sel, uk2, uk), jnp.where(sel, ck2, ck),
                    jnp.where(sel[None], jnp.concatenate([t1, t2]), traj))

        uk, ck, traj = jax.lax.cond(jnp.any(bad), redo, lambda _: (uk, ck, traj), None)
        return (uk, ck, n_rb + jnp.any(bad).astype(jnp.int32)), (traj, jnp.mean(e))

    return body


@partial(jax.jit, static_argnames=("grid", "k", "dt", "e_star", "n_records", "record_every"))
def cert_rollout(grid: Grid3, model: dict, u_hat, *, k: int, dt: float, e_star: float, n_records: int,
                 record_every: int):
    total = n_records * record_every
    body = _certified_body(grid, model, k, dt, e_star)
    init = (u_hat, correction(grid, model, u_hat), jnp.zeros((), jnp.int32))
    (_, _, n_rb), (traj, es) = jax.lax.scan(body, init, None, length=-(-total // k))
    flat = traj.reshape(-1, *traj.shape[2:])[:total]
    return flat[record_every - 1 :: record_every], n_rb, jnp.mean(es)


@partial(jax.jit, static_argnames=("grid", "k", "dt", "e_star", "n_steps"))
def cert_advance(grid: Grid3, model: dict, u_hat, *, k: int, dt: float, e_star: float, n_steps: int):
    body = _certified_body(grid, model, k, dt, e_star)
    init = (u_hat, correction(grid, model, u_hat), jnp.zeros((), jnp.int32))
    return jax.lax.scan(lambda c, x: (body(c, x)[0], None), init, None, length=n_steps // k)[0][0]
