"""Pseudo-spectral 3D incompressible Navier-Stokes: forced isotropic turbulence (workload W3).

    du/dt = P[u x omega] + nu lap u + f,   div u = 0,

on the periodic box [0, 2 pi)^3, in rotational form. The state is the rfftn of the three
velocity components, shape (..., 3, n, n, n//2 + 1), indexed ``u[..., c, i, j, l] =
u_c(x_i, y_j, z_l)``. P projects onto divergence-free fields.

- **Forcing** keeps the flow statistically stationary at a fixed energy input: the shells
  0 < |k| <= k_force are driven by f_hat = (power / (2 E_f)) u_hat there, so dE/dt from the
  forcing is exactly ``power`` (deterministic, as in band-limited constant-power forcing).
- **Time stepping** is Lawson's integrating-factor RK4, as in 2D: viscosity exactly, the
  nonlinear term, forcing and any correction explicitly.
- **Dealiasing**: the 2/3 rule, or (``dealias="exp"``) the exponential filter
  exp(-6 (|k| / (n/2))^32) applied after every step, the stronger classical setting
  found in 2D (new_directions.md S5).

Why 3D: Dresdner et al. (TMLR 2023) found little room for learned corrections against
spectral solvers in 2D, where the enstrophy cascade (E ~ k^-3) makes spectral methods
converge fast, and named 3D (E ~ k^-5/3) as where hybrids should pay. Resolution also
costs n^4 here (n^3 points, a time step ~ 1/n), against n^3 in 2D.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import jax
import jax.numpy as jnp
import numpy as np

AXES = (-3, -2, -1)


@dataclass(frozen=True)
class Grid3:
    """An n^3 periodic grid on [0, 2 pi)^3 and its rfftn wavenumbers."""

    n: int
    nu: float = 3e-3
    power: float = 0.1
    k_force: float = 2.0
    dealias: str = "two_thirds"  # or "exp"
    filter_strength: float = 6.0  # exp filter attenuation per step; 6 / m keeps the per-time damping at dt / m
    power_b: float = -1.0  # MHD (frost.cfd.mhd3d): power into the magnetic field; < 0 means ``power``, 0 none

    @cached_property
    def kx(self) -> np.ndarray:
        return (np.fft.fftfreq(self.n) * self.n).astype(np.float32)[:, None, None]

    @cached_property
    def ky(self) -> np.ndarray:
        return (np.fft.fftfreq(self.n) * self.n).astype(np.float32)[None, :, None]

    @cached_property
    def kz(self) -> np.ndarray:
        return (np.fft.rfftfreq(self.n) * self.n).astype(np.float32)[None, None, :]

    @cached_property
    def k2(self) -> np.ndarray:
        return self.kx**2 + self.ky**2 + self.kz**2

    @cached_property
    def kmax(self) -> int:
        if self.dealias not in ("two_thirds", "exp"):
            raise ValueError(self.dealias)
        return self.n // 3 if self.dealias == "two_thirds" else self.n // 2 - 1

    @cached_property
    def mask(self) -> np.ndarray:
        m = (np.abs(self.kx) <= self.kmax) & (np.abs(self.ky) <= self.kmax) & (self.kz <= self.kmax)
        return m.astype(np.float32)

    @cached_property
    def exp_filter(self) -> np.ndarray | None:
        if self.dealias == "two_thirds":
            return None
        return (np.exp(-self.filter_strength * (np.sqrt(self.k2) / (self.n / 2)) ** 32) * self.mask).astype(np.float32)

    @cached_property
    def inv_k2(self) -> np.ndarray:
        out = np.zeros_like(self.k2)
        np.divide(1.0, self.k2, out=out, where=self.k2 > 0)
        return out

    @cached_property
    def linear(self) -> np.ndarray:
        return (-self.nu * self.k2).astype(np.float32)

    @cached_property
    def forced(self) -> np.ndarray:
        return ((self.k2 > 0) & (np.sqrt(self.k2) <= self.k_force)).astype(np.float32)

    @cached_property
    def les_cutoff(self) -> int:
        """The cutoff k_c an LES closure sees: n/3 under the 2/3 rule; under the filter, where
        it reaches about 1/2 (0.47 n), floored to 0.45 n."""
        return self.n // 3 if self.dealias == "two_thirds" else int(0.45 * self.n)

    @cached_property
    def weight(self) -> np.ndarray:
        """rfftn half-space multiplicity: 1 on the kz = 0 plane, 2 elsewhere (Nyquist is masked)."""
        return np.where(self.kz == 0, 1.0, 2.0).astype(np.float32) * np.ones_like(self.k2)


def to_real(grid: Grid3, u_hat):
    return jnp.fft.irfftn(u_hat, s=(grid.n,) * 3, axes=AXES)


def to_hat(grid: Grid3, u):
    return jnp.fft.rfftn(u, axes=AXES) * grid.mask


def _k(grid: Grid3):
    return (grid.kx, grid.ky, grid.kz)


def project(grid: Grid3, f_hat):
    """Remove the gradient part: f - k (k . f) / |k|^2, over the component axis -4."""
    kx, ky, kz = _k(grid)
    kf = kx * f_hat[..., 0, :, :, :] + ky * f_hat[..., 1, :, :, :] + kz * f_hat[..., 2, :, :, :]
    kf = kf * grid.inv_k2
    return jnp.stack([f_hat[..., 0, :, :, :] - kx * kf, f_hat[..., 1, :, :, :] - ky * kf,
                      f_hat[..., 2, :, :, :] - kz * kf], axis=-4)


def curl(grid: Grid3, u_hat):
    kx, ky, kz = _k(grid)
    ux, uy, uz = u_hat[..., 0, :, :, :], u_hat[..., 1, :, :, :], u_hat[..., 2, :, :, :]
    return 1j * jnp.stack([ky * uz - kz * uy, kz * ux - kx * uz, kx * uy - ky * ux], axis=-4)


def energy_of(grid: Grid3, u_hat, where=None):
    """0.5 <|u|^2> from the rfftn coefficients (Parseval), per leading index."""
    w = grid.weight if where is None else grid.weight * where
    return 0.5 * jnp.sum(jnp.abs(u_hat) ** 2 * w, axis=(-4, -3, -2, -1)) / float(grid.n) ** 6


def forcing(grid: Grid3, u_hat):
    e_f = energy_of(grid, u_hat, grid.forced)
    scale = grid.power / (2.0 * jnp.maximum(e_f, 1e-12))
    return scale[..., None, None, None, None] * u_hat * grid.forced


def nonlinear(grid: Grid3, u_hat):
    """P[(u x omega)^], dealiased."""
    u = to_real(grid, u_hat)
    w = to_real(grid, curl(grid, u_hat))
    cross = jnp.stack([u[..., 1, :, :, :] * w[..., 2, :, :, :] - u[..., 2, :, :, :] * w[..., 1, :, :, :],
                       u[..., 2, :, :, :] * w[..., 0, :, :, :] - u[..., 0, :, :, :] * w[..., 2, :, :, :],
                       u[..., 0, :, :, :] * w[..., 1, :, :, :] - u[..., 1, :, :, :] * w[..., 0, :, :, :]],
                      axis=-4)
    return project(grid, to_hat(grid, cross))


def step(grid: Grid3, u_hat, dt: float, tendency=None):
    """One Lawson IFRK4 step; ``tendency(u_hat) -> hat`` is an extra explicit term (projected)."""

    def rhs(uh):
        out = nonlinear(grid, uh) + forcing(grid, uh)
        if tendency is not None:
            out = out + project(grid, tendency(uh) * grid.mask)
        return out

    e = jnp.exp(grid.linear * (0.5 * dt))
    e2 = e * e
    k1 = rhs(u_hat)
    k2 = rhs(e * (u_hat + 0.5 * dt * k1))
    k3 = rhs(e * u_hat + 0.5 * dt * k2)
    k4 = rhs(e2 * u_hat + dt * e * k3)
    out = e2 * u_hat + (dt / 6.0) * (e2 * k1 + 2.0 * e * (k2 + k3) + k4)
    return out if grid.exp_filter is None else out * grid.exp_filter


# ------------------------------------------------------------------ classical LES closures
def chollet_lesieur(grid: Grid3, c: float = 1.0, ck: float = 1.4):
    """Chollet-Lesieur spectral eddy viscosity (1981), the classical closure for spectral LES of
    isotropic turbulence: nu_t(k) = c 0.441 C_K^(-3/2) (1 + 34.5 exp(-3.03 k_c / k)) sqrt(E(k_c) / k_c),
    with E(k_c) the energy in the shell at the cutoff. Returns ``tendency(u_hat)`` for ``step``."""
    kc = grid.les_cutoff
    k = np.sqrt(grid.k2)
    cusp = np.where(k > 0, 1.0 + 34.5 * np.exp(-3.03 * kc / np.maximum(k, 1e-6)), 0.0)
    kernel = (cusp * grid.k2).astype(np.float32)
    shell = ((k >= kc - 0.5) & (k < kc + 0.5)).astype(np.float32)
    coef = c * 0.441 * ck**-1.5

    def tendency(u_hat):
        nu = coef * jnp.sqrt(jnp.maximum(energy_of(grid, u_hat, shell), 0.0) / kc)
        return -nu[..., None, None, None, None] * kernel * u_hat

    return tendency


def smagorinsky(grid: Grid3, cs: float = 0.17):
    """Smagorinsky (1963): f = div(2 nu_t S), nu_t = (c_s Delta)^2 |S|, Delta = pi / k_c, with the
    gradients taken spectrally. Returns ``tendency(u_hat)`` for ``step``."""
    kvec = _k(grid)
    nu_scale = (cs * np.pi / grid.les_cutoff) ** 2

    def tendency(u_hat):
        grads = 1j * jnp.stack([u_hat * kj for kj in kvec], axis=-4)   # (..., 3 i, 3 j, n, n, m): d_j u_i
        g = to_real(grid, grads)
        s_ij = 0.5 * (g + jnp.swapaxes(g, -4, -5))
        s_norm = jnp.sqrt(2.0 * jnp.sum(s_ij**2, axis=(-5, -4)))
        tau = to_hat(grid, 2.0 * nu_scale * s_norm[..., None, None, :, :, :] * s_ij)
        return sum(1j * kvec[j] * tau[..., j, :, :, :, :] for j in range(3))  # sum_j d_j tau_ji (tau symmetric)

    return tendency


def mixed(grid: Grid3, cs: float = 0.1, cg: float = 1.0 / 12.0):
    """Mixed model: Smagorinsky plus the Clark gradient model, f = div(2 nu_t S - tau^C) with
    tau^C_ij = cg Delta^2 d_k u_i d_k u_j (Clark, Ferziger and Reynolds 1979; cg = 1/12 for a box
    filter). Eddy viscosity alone is purely dissipative and makes an LES's gradient statistics those
    of a lower-Re flow; the gradient term carries the nonlinear, non-dissipative transfer that keeps
    the resolved strain skewed as in filtered DNS (arXiv 2504.19356). The Smagorinsky part keeps it
    stable (the gradient model alone is not; Nabavi and Kim 2024)."""
    kvec = _k(grid)
    delta2 = (np.pi / grid.les_cutoff) ** 2

    def tendency(u_hat):
        grads = 1j * jnp.stack([u_hat * kj for kj in kvec], axis=-4)   # (..., 3 i, 3 j, n, n, m): d_j u_i
        g = to_real(grid, grads)
        s_ij = 0.5 * (g + jnp.swapaxes(g, -4, -5))
        s_norm = jnp.sqrt(2.0 * jnp.sum(s_ij**2, axis=(-5, -4)))
        clark = jnp.einsum("...ikxyz,...jkxyz->...ijxyz", g, g)
        stress = 2.0 * cs**2 * delta2 * s_norm[..., None, None, :, :, :] * s_ij - cg * delta2 * clark
        tau = to_hat(grid, stress)
        return sum(1j * kvec[j] * tau[..., j, :, :, :, :] for j in range(3))  # sum_j d_j T_ji (T symmetric)

    return tendency


def les_tendency(grid: Grid3, kind: str, c: float):
    """``none`` | ``cl`` (Chollet-Lesieur, c scales nu_t) | ``smag`` (Smagorinsky, c = c_s) |
    ``mixed`` (Smagorinsky with c_s = c plus the Clark gradient model)."""
    if kind == "none":
        return None
    if kind == "cl":
        return chollet_lesieur(grid, c)
    if kind == "smag":
        return smagorinsky(grid, c)
    if kind == "mixed":
        return mixed(grid, c)
    raise ValueError(kind)


# ------------------------------------------------------------------ grids, fields, diagnostics
def truncate(fine: Grid3, coarse: Grid3, u_hat):
    """Sharp spectral filter from ``fine`` to ``coarse`` (then the coarse filter, if any)."""
    nf, nc, kc = fine.n, coarse.n, coarse.kmax
    scale = (nc / nf) ** 3
    lead = u_hat.shape[:-3]
    out = jnp.zeros((*lead, nc, nc, nc // 2 + 1), dtype=u_hat.dtype)
    for xs_c, xs_f in ((slice(0, kc + 1), slice(0, kc + 1)), (slice(nc - kc, nc), slice(nf - kc, nf))):
        for ys_c, ys_f in ((slice(0, kc + 1), slice(0, kc + 1)), (slice(nc - kc, nc), slice(nf - kc, nf))):
            out = out.at[..., xs_c, ys_c, : kc + 1].set(u_hat[..., xs_f, ys_f, : kc + 1] * scale)
    out = out * coarse.mask
    return out if coarse.exp_filter is None else out * coarse.exp_filter


def pad(coarse: Grid3, fine: Grid3, u_hat):
    """Zero-pad a coarse rfftn spectrum onto a finer grid: the inverse of ``truncate`` on the coarse band."""
    nf, nc, kc = fine.n, coarse.n, coarse.kmax
    scale = (nf / nc) ** 3
    out = jnp.zeros((*u_hat.shape[:-3], nf, nf, nf // 2 + 1), dtype=u_hat.dtype)
    for xs_f, xs_c in ((slice(0, kc + 1), slice(0, kc + 1)), (slice(nf - kc, nf), slice(nc - kc, nc))):
        for ys_f, ys_c in ((slice(0, kc + 1), slice(0, kc + 1)), (slice(nf - kc, nf), slice(nc - kc, nc))):
            out = out.at[..., xs_f, ys_f, : kc + 1].set(u_hat[..., xs_c, ys_c, : kc + 1] * scale)
    return out


def random_initial(grid: Grid3, key, peak: float = 3.0, u_rms: float = 0.5, batch=()):
    """A random divergence-free field with an energy spectrum peaked near ``peak``."""
    shape = (*batch, 3, grid.n, grid.n, grid.n // 2 + 1)
    kr, ki = jax.random.split(key)
    phase = jax.random.normal(kr, shape) + 1j * jax.random.normal(ki, shape)
    k = jnp.sqrt(grid.k2)
    amp = k * jnp.exp(-((k - peak) ** 2) / 2.0) * grid.mask
    u_hat = project(grid, phase * amp)
    u_hat = to_hat(grid, to_real(grid, u_hat))  # Hermitian-consistent kz = 0 plane
    u_hat = project(grid, u_hat)
    e = energy_of(grid, u_hat)
    return u_hat * (jnp.sqrt(1.5 * u_rms**2 / e))[..., None, None, None, None]


def dissipation(grid: Grid3, u_hat):
    """nu <|omega|^2> = 2 nu sum k^2 E(k), per leading index."""
    return 2.0 * grid.nu * energy_of(grid, curl(grid, u_hat))


def divergence_max(grid: Grid3, u_hat):
    kx, ky, kz = _k(grid)
    d = 1j * (kx * u_hat[..., 0, :, :, :] + ky * u_hat[..., 1, :, :, :] + kz * u_hat[..., 2, :, :, :])
    return jnp.max(jnp.abs(to_real(grid, d[..., None, :, :, :])))


def correlation(a, b):
    """Pearson correlation of two real vector fields over the last four axes (3, n, n, n)."""
    a = a - a.mean(axis=(-3, -2, -1), keepdims=True)
    b = b - b.mean(axis=(-3, -2, -1), keepdims=True)
    ax = (-4, -3, -2, -1)
    return (a * b).sum(axis=ax) / jnp.sqrt((a * a).sum(axis=ax) * (b * b).sum(axis=ax))


def max_speed(grid: Grid3, u_hat):
    u = to_real(grid, u_hat)
    return jnp.max(jnp.sqrt(jnp.sum(u**2, axis=-4)))
