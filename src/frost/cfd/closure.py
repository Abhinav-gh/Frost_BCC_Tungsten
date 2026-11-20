"""Learned and classical closures for workload W1, and the ways a runtime may hold them.

The learned closure is a CNN that maps the coarse state (u, v, omega) to a *local
flux operator*: at every cell, 3x3 stencil coefficients for each flux component,

    q_x(x) = sum_s a^x_s(x) omega(x + s),   q_y likewise,   tendency = -div q.

The divergence form conserves mean vorticity exactly. Writing the closure as
coefficients applied to the state (the structure of Kochkov et al.'s learned
interpolation) exposes two cut points for reuse (new_directions.md S1):
- **value**: hold the tendency -div q computed at the refresh;
- **operator**: hold the coefficients a and apply them to the *fresh* state.

The unconstrained stencil can be anti-diffusive at the grid scale, and training
through 16-step unrolls blew it up (test 11, 2026-10-01). The default form is
therefore **structured** (``form="tensor"``), with four outputs per cell:

    tendency = div(K grad omega) + J(s, omega),

- K = [[k11, k12], [k12, k22]] symmetric positive semi-definite: softplus
  diagonals, and k12 = rho sqrt(k11 k22) with |rho| < 1. It can only remove
  enstrophy, and is anisotropic eddy diffusion.
- s is a stream function whose advection J(s, omega) = -div(omega curl s)
  conserves enstrophy exactly (spectral derivatives): a learned subgrid advection,
  which allows organised transfer without injecting enstrophy.

The held operator is (k11, k12, k22, curl s), applied to the fresh state.

The network is our own on ``jax.lax.conv_general_dilated`` with circular padding.
The last layer starts at zero, so an untrained closure is no closure.

Smagorinsky (eddy viscosity (C_s Delta)^2 |S| acting on omega) is the cheap
student for the residual hold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.ndimage import map_coordinates

from frost.cfd.spectral import Grid, ddx, ddy, to_hat, to_real, velocity

@dataclass(frozen=True)
class Static:
    """Model settings that select code paths: hashed into the treedef, never traced."""

    base: str = "central"   # finite-volume interpolation the learned correction sits on
    inplane: bool = False   # MHD: correct only E_z (2D in-plane field)


jax.tree_util.register_static(Static)

OFFSETS = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)]
N_COEF = 2 * len(OFFSETS)  # the stencil form
N_RAW = {"stencil": N_COEF, "tensor": 4}
K_OFFSET = -2.0  # softplus(-2) = 0.13: the untrained closure is a weak isotropic diffusion


# ------------------------------------------------------------------ the network
def init_cnn(key, in_ch: int = 3, hidden: int = 64, layers: int = 6, out_ch: int = N_COEF,
             kernel: int = 3) -> list[dict]:
    chans = [in_ch] + [hidden] * (layers - 1) + [out_ch]
    keys = jax.random.split(key, layers)
    params = []
    for i, (a, b) in enumerate(zip(chans[:-1], chans[1:])):
        w = jax.random.normal(keys[i], (kernel, kernel, a, b)) * math.sqrt(2.0 / (a * kernel * kernel))
        if i == layers - 1:
            w = jnp.zeros_like(w)
        params.append({"w": w.astype(jnp.float32), "b": jnp.zeros((b,), jnp.float32)})
    return params


def apply_cnn(params: list[dict], x, padding: str = "wrap"):
    """x: (B, n, n, C) -> (B, n, n, out), periodic.

    ``padding="valid"`` applies no padding: a patch carrying a halo of one cell per
    layer comes out as its centre (``frost.cfd.tiles``).
    """
    pad = params[0]["w"].shape[0] // 2
    for i, p in enumerate(params):
        if padding == "wrap":
            x = jnp.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="wrap")
        x = jax.lax.conv_general_dilated(x, p["w"], (1, 1), "VALID",
                                         dimension_numbers=("NHWC", "HWIO", "NHWC")) + p["b"]
        if i < len(params) - 1:
            x = jax.nn.gelu(x)
    return x


def cnn_flops(params: list[dict], n: int) -> float:
    """Multiply-add FLOPs of one call on an n x n field (2 per MAC)."""
    return float(sum(2 * n * n * np.prod(p["w"].shape) for p in params))


# ------------------------------------------------------------------ the learned closure
def features(grid: Grid, w_hat, scale: dict):
    """(B, n, n, 3): u, v, omega in real space, normalised."""
    u, v = velocity(grid, w_hat)
    w = to_real(grid, w_hat)
    return jnp.stack([u / scale["u"], v / scale["u"], w / scale["w"]], axis=-1)


def coef_from_raw(grid: Grid, model: dict, raw):
    """Network output -> the held operator. Cheap: elementwise, plus two FFTs for curl s."""
    if raw.shape[-1] == N_COEF:
        bound = model["scale"].get("bound")
        if bound is not None:  # bounded stencil: a held operator cannot grow without limit
            return bound * jnp.tanh(raw) * model["scale"]["u"]
        return raw * model["scale"]["u"]
    sc = model["scale"]
    k11 = sc["k"] * jax.nn.softplus(raw[..., 0] + K_OFFSET)
    k22 = sc["k"] * jax.nn.softplus(raw[..., 1] + K_OFFSET)
    k12 = 0.99 * jnp.tanh(raw[..., 2]) * jnp.sqrt(k11 * k22)
    s_hat = to_hat(grid, sc["s"] * raw[..., 3])
    usx = to_real(grid, ddy(grid, s_hat))   # curl s = (ds/dy, -ds/dx): divergence-free
    usy = to_real(grid, -ddx(grid, s_hat))
    return jnp.stack([k11, k12, k22, usx, usy], axis=-1)


def coefficients(grid: Grid, model: dict, w_hat):
    """The expensive part: one CNN call, then the cheap transform to the held operator."""
    return coef_from_raw(grid, model, apply_cnn(model["params"], features(grid, w_hat, model["scale"])))


def apply_operator(grid: Grid, coef, w_hat):
    """The cheap part: the held operator applied to the given (fresh) state."""
    if coef.shape[-1] == 5:
        return _apply_tensor(grid, coef, w_hat)
    w = to_real(grid, w_hat)
    m = len(OFFSETS)
    qx = jnp.zeros_like(w)
    qy = jnp.zeros_like(w)
    for s, (di, dj) in enumerate(OFFSETS):
        shifted = jnp.roll(w, shift=(-di, -dj), axis=(-2, -1))  # omega(x + s)
        qx = qx + coef[..., s] * shifted
        qy = qy + coef[..., m + s] * shifted
    return -(ddx(grid, to_hat(grid, qx)) + ddy(grid, to_hat(grid, qy)))


def _apply_tensor(grid: Grid, coef, w_hat):
    """div(K grad omega - omega curl s): dissipative plus enstrophy-conserving."""
    w = to_real(grid, w_hat)
    wx = to_real(grid, ddx(grid, w_hat))
    wy = to_real(grid, ddy(grid, w_hat))
    k11, k12, k22, usx, usy = (coef[..., i] for i in range(5))
    qx = k11 * wx + k12 * wy - w * usx
    qy = k12 * wx + k22 * wy - w * usy
    return ddx(grid, to_hat(grid, qx)) + ddy(grid, to_hat(grid, qy))


def smagorinsky(grid: Grid, w_hat, cs: float = 0.17):
    """div((C_s Delta)^2 |S| grad omega), Delta = pi / kmax (the sharp filter's width)."""
    psi = w_hat * grid.inv_k2
    ux = to_real(grid, ddx(grid, ddy(grid, psi)))       # du/dx = d2psi/dxdy
    uy = to_real(grid, ddy(grid, ddy(grid, psi)))       # du/dy
    vx = to_real(grid, -ddx(grid, ddx(grid, psi)))      # dv/dx
    s_mag = jnp.sqrt(2.0 * (ux**2 + (-ux) ** 2 + 0.5 * (uy + vx) ** 2))
    nu_t = (cs * math.pi / grid.kmax) ** 2 * s_mag
    qx = nu_t * to_real(grid, ddx(grid, w_hat))
    qy = nu_t * to_real(grid, ddy(grid, w_hat))
    return ddx(grid, to_hat(grid, qx)) + ddy(grid, to_hat(grid, qy))


# ------------------------------------------------------------------ transport of a held field
def advect(grid: Grid, field, w_hat, dt: float):
    """Semi-Lagrangian transport of a held real field (B, n, n, C) by one step.

    field(x) <- field(x - u dt), bilinear and periodic. This holds the cached
    quantity in the frame of the flow, as MD's per-atom residual is held in the
    frame of its atom.
    """
    u, v = velocity(grid, w_hat)
    n = grid.n
    ii, jj = jnp.meshgrid(jnp.arange(n, dtype=jnp.float32), jnp.arange(n, dtype=jnp.float32),
                          indexing="ij")
    src_i = ii - u * dt / grid.dx
    src_j = jj - v * dt / grid.dx

    def one(f, si, sj):
        return map_coordinates(f, [si, sj], order=1, mode="wrap")

    per_channel = jax.vmap(one, in_axes=(-1, None, None), out_axes=-1)
    return jax.vmap(per_channel)(field, src_i, src_j)
