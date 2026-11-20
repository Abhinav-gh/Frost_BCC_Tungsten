"""Passive tracers carried by a Kolmogorov flow (workload W1-T).

The velocity forecast of W1 is lost to the spectral solver by the initial-condition
ceiling (``experiments/e10_cfd/ceiling.py``). Tracers are the other half of most
transport workloads (chemistry, salinity, dye): a smooth initial field is stretched into
filaments thinner than the grid, so the coarse-grid answer depends on the scheme, not
on information the initial state lacks. This module holds the classical schemes a
learned one must beat:

    dc/dt + div(u c) = kappa lap c,     div u = 0,

on the periodic box [0, 2 pi)^2, indexed ``c[..., i, j] = c(x_i, y_j)`` as in
``frost.cfd.spectral``. The velocity comes from a vorticity spectrum on a coarser grid
(the flow solver's), padded onto the tracer grid; every scheme sees the same velocity.

Schemes:
- **spectral** (point values): conservative form, Lawson IFRK4 with diffusion exact,
  dealiased by the grid's 2/3 rule or exponential filter. Not positivity-preserving;
  ``hole_fill`` restores positivity at the points by clipping, mass-conserving.
- **finite volume** (cell averages), flux form, SSP-RK3, 5-point Laplacian:
  - ``vanleer``: MUSCL with the van Leer limiter, positive for CFL <= 1/4 (unsplit);
  - ``weno5z``: fifth-order WENO-Z reconstruction (Borges et al. 2008), not positive;
  - ``weno5z_pp``: WENO-Z with a positivity-preserving flux limiter (``fv_euler``);
  - ``li_pp``: WENO-Z plus a learned stencil correction (Kochkov et al. 2021, Zhuang et
    al. 2021), under the same limiter. Its coefficients are the expensive part: one CNN
    call per field, held over SSP-RK3 stages, substeps and, under Frost, k intervals.
  The face velocities are differences of the stream function at cell corners, i.e.
  the exact face-integrated normal velocity, so the discrete divergence is zero to
  round-off and the mass is conserved.
"""

from __future__ import annotations

from functools import cached_property

import jax
import jax.numpy as jnp
import numpy as np

from frost.cfd import closure, spectral
from frost.cfd import finite_volume as fv
from frost.cfd.spectral import Grid

FV_SCHEMES = ("vanleer", "weno5z", "weno5z_pp")  # the classical ones; li_pp needs a model


# ------------------------------------------------------------------ velocity
def corner_shift(grid: Grid) -> np.ndarray:
    """The phase that moves a spectrum by half a cell in x and y (values at cell corners)."""
    return np.exp(0.5j * grid.dx * (grid.kx + grid.ky)).astype(np.complex64)


def face_velocity(flow: Grid, grid: Grid, w_hat):
    """Face-integrated normal velocities on ``grid`` from vorticity ``w_hat`` on ``flow``.

    ``ux[i, j]`` is the mean of u over the face (i+1/2, j), ``vy[i, j]`` that of v over
    (i, j+1/2); both are differences of psi at the corners, so the discrete divergence
    vanishes identically.
    """
    w = w_hat if flow == grid else spectral.pad(flow, grid, w_hat)
    psi_c = spectral.to_real(grid, w * grid.inv_k2 * corner_shift(grid))  # psi(x_i + dx/2, y_j + dx/2)
    ux = (psi_c - jnp.roll(psi_c, 1, axis=-1)) / grid.dx
    vy = -(psi_c - jnp.roll(psi_c, 1, axis=-2)) / grid.dx
    return ux, vy


def point_velocity(flow: Grid, grid: Grid, w_hat):
    """(u, v) at the points of ``grid`` (for the spectral scheme)."""
    return spectral.velocity(grid, w_hat if flow == grid else spectral.pad(flow, grid, w_hat))


# ------------------------------------------------------------------ spectral scheme
def spectral_step(grid: Grid, c_hat, u, v, dt: float, kappa: float):
    """One Lawson IFRK4 step of the tracer in conservative form."""

    def rhs(ch):
        c = spectral.to_real(grid, ch)
        return -(spectral.ddx(grid, spectral.to_hat(grid, u * c)) + spectral.ddy(grid, spectral.to_hat(grid, v * c)))

    e = jnp.exp(-kappa * grid.k2 * (0.5 * dt))
    e2 = e * e
    k1 = rhs(c_hat)
    k2 = rhs(e * (c_hat + 0.5 * dt * k1))
    k3 = rhs(e * c_hat + 0.5 * dt * k2)
    k4 = rhs(e2 * c_hat + dt * e * k3)
    out = e2 * c_hat + (dt / 6.0) * (e2 * k1 + 2.0 * e * (k2 + k3) + k4)
    return out if grid.exp_filter is None else out * grid.exp_filter


def hole_fill(grid: Grid, c_hat):
    """Positivity for the spectral scheme, as spectral transport models did it: clip the
    negative points and rescale the rest so the mass is unchanged."""
    c = spectral.to_real(grid, c_hat)
    pos = jnp.maximum(c, 0.0)
    pos = pos * (c.mean(axis=(-2, -1), keepdims=True) / pos.mean(axis=(-2, -1), keepdims=True))
    return jnp.fft.rfft2(pos)


# ------------------------------------------------------------------ finite-volume schemes
def _sh(c, s: int, axis: int):
    """c shifted so that index i holds c[i + s] along ``axis`` (periodic)."""
    return jnp.roll(c, -s, axis=axis)


def _vanleer_slope(a, b):
    return jnp.where(a * b > 0.0, 2.0 * a * b / jnp.where(a + b == 0.0, 1.0, a + b), 0.0)


def face_vanleer(c, vel, axis: int):
    """Upwinded MUSCL face value at i+1/2 along ``axis``, van Leer limiter."""
    cm, cp, cpp = _sh(c, -1, axis), _sh(c, 1, axis), _sh(c, 2, axis)
    s0 = _vanleer_slope(c - cm, cp - c)
    s1 = _vanleer_slope(cp - c, cpp - cp)
    return jnp.where(vel >= 0.0, c + 0.5 * s0, cp - 0.5 * s1)


def _weno5z(a, b, c, d, e, eps: float = 1e-12):
    """WENO-Z value at the right face of cell c, from cells a..e (left-biased)."""
    q0 = (2.0 * a - 7.0 * b + 11.0 * c) / 6.0
    q1 = (-b + 5.0 * c + 2.0 * d) / 6.0
    q2 = (2.0 * c + 5.0 * d - e) / 6.0
    b0 = 13.0 / 12.0 * (a - 2.0 * b + c) ** 2 + 0.25 * (a - 4.0 * b + 3.0 * c) ** 2
    b1 = 13.0 / 12.0 * (b - 2.0 * c + d) ** 2 + 0.25 * (b - d) ** 2
    b2 = 13.0 / 12.0 * (c - 2.0 * d + e) ** 2 + 0.25 * (3.0 * c - 4.0 * d + e) ** 2
    tau = jnp.abs(b0 - b2)
    a0 = 0.1 * (1.0 + (tau / (b0 + eps)) ** 2)
    a1 = 0.6 * (1.0 + (tau / (b1 + eps)) ** 2)
    a2 = 0.3 * (1.0 + (tau / (b2 + eps)) ** 2)
    return (a0 * q0 + a1 * q1 + a2 * q2) / (a0 + a1 + a2)


def face_weno5z(c, vel, axis: int):
    """Upwinded WENO-Z face value at i+1/2 along ``axis``."""
    m2, m1, p1, p2, p3 = (_sh(c, s, axis) for s in (-2, -1, 1, 2, 3))
    left = _weno5z(m2, m1, c, p1, p2)
    right = _weno5z(p3, p2, p1, c, m1)
    return jnp.where(vel >= 0.0, left, right)


FACE = {"vanleer": face_vanleer, "weno5z": face_weno5z}


def _div(fx, fy, dx: float):
    """Divergence of face fluxes (fx at i+1/2 along axis -2, fy at j+1/2 along axis -1)."""
    return ((fx - jnp.roll(fx, 1, axis=-2)) + (fy - jnp.roll(fy, 1, axis=-1))) / dx


def faces(scheme: str, c, ux, vy, coef=None):
    """Face values (x-faces, y-faces) of a scheme; ``li`` is WENO-Z plus a learned stencil."""
    if scheme == "li":
        return li_faces(coef, c, ux, vy)
    return FACE[scheme](c, ux, -2), FACE[scheme](c, vy, -1)


def fv_euler(scheme: str, dx: float, c, ux, vy, dt: float, kappa: float, coef=None):
    """One forward-Euler stage of -div(u c) + kappa lap c on cell averages.

    A ``_pp`` scheme limits its fluxes toward first-order upwind so the stage stays
    non-negative: the positivity half of Zalesak's flux-corrected transport. Each
    cell's outgoing antidiffusive flux is scaled by the share its low-order update can
    pay for; faces take their donor cell's factor, so mass is conserved exactly and an
    unthreatened cell keeps the full high-order flux.
    """
    cx, cy = faces(scheme.removesuffix("_pp"), c, ux, vy, coef)
    fx, fy = ux * cx, vy * cy
    if kappa:
        dfx, dfy = -kappa * (jnp.roll(c, -1, -2) - c) / dx, -kappa * (jnp.roll(c, -1, -1) - c) / dx
        fx, fy = fx + dfx, fy + dfy
    if not scheme.endswith("_pp"):
        return c - dt * _div(fx, fy, dx)
    lx = ux * jnp.where(ux >= 0.0, c, jnp.roll(c, -1, -2))
    ly = vy * jnp.where(vy >= 0.0, c, jnp.roll(c, -1, -1))
    if kappa:
        lx, ly = lx + dfx, ly + dfy
    low = c - dt * _div(lx, ly, dx)
    ax, ay = fx - lx, fy - ly
    out = (jnp.maximum(ax, 0.0) + jnp.maximum(-jnp.roll(ax, 1, -2), 0.0) + jnp.maximum(ay, 0.0)
           + jnp.maximum(-jnp.roll(ay, 1, -1), 0.0)) * (dt / dx)
    r = jnp.where(out > 0.0, jnp.minimum(1.0, jnp.maximum(low, 0.0) / jnp.where(out > 0.0, out, 1.0)), 1.0)
    tx = jnp.where(ax >= 0.0, r, jnp.roll(r, -1, -2))
    ty = jnp.where(ay >= 0.0, r, jnp.roll(r, -1, -1))
    return low - dt * _div(tx * ax, ty * ay, dx)


def fv_step(scheme: str, dx: float, c, ux, vy, dt: float, kappa: float, coef=None):
    """One SSP-RK3 step (Shu and Osher): convex combinations of Euler stages, so a
    positive stage gives a positive step. A learned scheme's ``coef`` is held over it."""
    c1 = fv_euler(scheme, dx, c, ux, vy, dt, kappa, coef)
    c2 = 0.75 * c + 0.25 * fv_euler(scheme, dx, c1, ux, vy, dt, kappa, coef)
    return c / 3.0 + 2.0 / 3.0 * fv_euler(scheme, dx, c2, ux, vy, dt, kappa, coef)


# ------------------------------------------------------------------ learned interpolation
def li_features(c, ux, vy, scale: dict):
    """(tracer, cell-centre u, v), normalised, channels last; the velocity broadcast over tracers."""
    uc = jnp.broadcast_to(0.5 * (ux + jnp.roll(ux, 1, -2)), c.shape)
    vc = jnp.broadcast_to(0.5 * (vy + jnp.roll(vy, 1, -1)), c.shape)
    return jnp.stack([(c - 1.0) / scale["c"], uc / scale["u"], vc / scale["u"]], axis=-1)


def li_coefficients(model: dict, c, ux, vy):
    """The expensive part: one CNN call per tracer field. (..., n, n) -> (..., n, n, 24)."""
    x = li_features(c, ux, vy, model["scale"])
    lead = x.shape[:-3]
    raw = closure.apply_cnn(model["params"], x.reshape(-1, *x.shape[-3:]))
    return fv.li_from_raw(raw, model["scale"].get("bound", 0.5)).reshape(*lead, *raw.shape[-3:])


def li_faces(coef, c, ux, vy):
    """WENO-Z face values plus the learned zero-sum stencil correction (Kochkov's constrained form)."""
    cx, cy = face_weno5z(c, ux, -2), face_weno5z(c, vy, -1)
    m = coef.shape[-1] // 2
    for s, (a, b) in enumerate(fv.X_STENCIL):
        cx = cx + coef[..., s] * fv.sh(c, a, b)
    for s, (a, b) in enumerate(fv.Y_STENCIL):
        cy = cy + coef[..., m + s] * fv.sh(c, a, b)
    return cx, cy


# ------------------------------------------------------------------ representations
class CellAverage:
    """Conversion between point-value spectra and cell averages on one grid.

    The cell average of exp(i k.x) is sinc(kx dx / 2) sinc(ky dx / 2) exp(i k.x).
    """

    def __init__(self, grid: Grid):
        self.grid = grid

    @cached_property
    def factor(self) -> np.ndarray:
        h = 0.5 * self.grid.dx
        return (np.sinc(self.grid.kx * h / np.pi) * np.sinc(self.grid.ky * h / np.pi)).astype(np.float32)

    def from_points(self, c_hat):
        return spectral.to_real(self.grid, c_hat * self.factor)

    def to_points(self, c):
        return jnp.fft.rfft2(c) / self.factor


def band_mask(grid: Grid, kc: int) -> np.ndarray:
    """The square band |kx|, |ky| <= kc on ``grid``'s rfft2 layout (a sharp projection)."""
    return ((np.abs(grid.kx) <= kc) & (grid.ky <= kc)).astype(np.float32)


def truncate_sharp(grid: Grid, n: int, kc: int, c_hat):
    """The band |kx|, |ky| <= kc of a spectrum on ``grid``, on an n^2 rfft2 layout (no filter)."""
    scale = (n / grid.n) ** 2
    out = jnp.zeros((*c_hat.shape[:-2], n, n // 2 + 1), dtype=c_hat.dtype)
    out = out.at[..., : kc + 1, : kc + 1].set(c_hat[..., : kc + 1, : kc + 1] * scale)
    return out.at[..., n - kc :, : kc + 1].set(c_hat[..., grid.n - kc :, : kc + 1] * scale)


def random_tracer(grid: Grid, key, kmax: int = 4, batch=()):
    """A non-negative tracer of mean 1 with zeros: g^2 for a random field g on |k| <= kmax.

    Band-limited to |k| <= 2 kmax, so every grid here holds it exactly.
    """
    shape = (*batch, grid.n, grid.n // 2 + 1)
    kr, ki = jax.random.split(key)
    g_hat = (jax.random.normal(kr, shape) + 1j * jax.random.normal(ki, shape)) * (grid.k2 <= kmax**2)
    g_hat = g_hat.at[..., 0, 0].set(0.0)
    g = spectral.to_real(grid, g_hat)
    c = g * g
    c = c / c.mean(axis=(-2, -1), keepdims=True)
    return jnp.fft.rfft2(c)
