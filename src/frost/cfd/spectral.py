"""Pseudo-spectral 2D incompressible Navier-Stokes in vorticity form (workload W1).

The Kolmogorov-flow benchmark of Kochkov et al. (PNAS 2021), on the periodic box
[0, 2 pi)^2:

    d omega/dt + u . grad omega = nu lap omega - alpha omega + F,
    F = -k_f cos(k_f y)        (the curl of the forcing sin(k_f y) x-hat),

with nu = 1e-3 (Re = 1000), alpha = 0.1 and k_f = 4. Arrays are indexed
``field[..., i, j] = f(x_i, y_j)``; the state is the rfft2 of omega over the last two
axes, kept inside the 2/3-rule band. Time stepping is Lawson's integrating-factor
RK4: diffusion and drag exactly, advection, forcing and any closure tendency
explicitly.

``Grid(n, dealias="exp")`` replaces the 2/3 rule by the exponential filter
exp(-6 (|k| / (n/2))^32) applied to the state after every step (Hou and Li 2007; the
setting of Dresdner et al. 2023). It keeps a disc of radius ~0.45 n instead of a square
of half-width n/3, so the same grid resolves more of the flow; it is the stronger
classical baseline.

A closure enters as a *tendency* added to the explicit part (the form INC,
NeurIPS 2025, recommends over state updates). ``step`` takes it as a function of the
stage state, so a held value (constant) and a held operator (applied to each fresh
stage) use the same integrator.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import jax
import jax.numpy as jnp
import numpy as np

TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class Grid:
    """An n x n periodic grid on [0, 2 pi)^2 and its rfft2 wavenumbers."""

    n: int
    nu: float = 1e-3
    alpha: float = 0.1
    k_force: int = 4
    dealias: str = "two_thirds"  # or "exp"

    @property
    def dx(self) -> float:
        return TWO_PI / self.n

    @cached_property
    def kx(self) -> np.ndarray:
        return (np.fft.fftfreq(self.n) * self.n).astype(np.float32)[:, None]

    @cached_property
    def ky(self) -> np.ndarray:
        return (np.fft.rfftfreq(self.n) * self.n).astype(np.float32)[None, :]

    @cached_property
    def k2(self) -> np.ndarray:
        return self.kx**2 + self.ky**2

    @cached_property
    def kmax(self) -> int:
        """Largest wavenumber kept: n/3 by the 2/3 rule, n/2 - 1 (Nyquist dropped) under the filter."""
        if self.dealias not in ("two_thirds", "exp"):
            raise ValueError(self.dealias)
        return self.n // 3 if self.dealias == "two_thirds" else self.n // 2 - 1

    @cached_property
    def exp_filter(self) -> np.ndarray | None:
        """exp(-6 (|k| / (n/2))^32) on the kept band, or None under the 2/3 rule."""
        if self.dealias == "two_thirds":
            return None
        return (np.exp(-6.0 * (np.sqrt(self.k2) / (self.n / 2)) ** 32) * self.mask).astype(np.float32)

    @cached_property
    def mask(self) -> np.ndarray:
        return ((np.abs(self.kx) <= self.kmax) & (self.ky <= self.kmax)).astype(np.float32)

    @cached_property
    def inv_k2(self) -> np.ndarray:
        out = np.zeros_like(self.k2)
        np.divide(1.0, self.k2, out=out, where=self.k2 > 0)
        return out

    @cached_property
    def linear(self) -> np.ndarray:
        """The implicit part, -nu k^2 - alpha, per mode."""
        return (-self.nu * self.k2 - self.alpha).astype(np.float32)

    @cached_property
    def coords(self) -> tuple[np.ndarray, np.ndarray]:
        x = (np.arange(self.n) * self.dx).astype(np.float32)
        return x[:, None], x[None, :]

    @cached_property
    def forcing_hat(self) -> np.ndarray:
        # numpy, not jnp: a cached jnp value first touched inside a trace would leak a tracer
        _, y = self.coords
        f = -self.k_force * np.cos(self.k_force * y) * np.ones((self.n, 1))
        return (np.fft.rfft2(f) * self.mask).astype(np.complex64)


def to_real(grid: Grid, w_hat):
    return jnp.fft.irfft2(w_hat, s=(grid.n, grid.n))


def to_hat(grid: Grid, w):
    return jnp.fft.rfft2(w) * grid.mask


def ddx(grid: Grid, f_hat):
    return 1j * grid.kx * f_hat


def ddy(grid: Grid, f_hat):
    return 1j * grid.ky * f_hat


def velocity(grid: Grid, w_hat):
    """(u, v) in real space from vorticity: psi = omega / k^2, u = dpsi/dy, v = -dpsi/dx."""
    psi = w_hat * grid.inv_k2
    return to_real(grid, ddy(grid, psi)), to_real(grid, -ddx(grid, psi))


def advection_hat(grid: Grid, w_hat):
    """-(u . grad omega), dealiased, in spectral space."""
    u, v = velocity(grid, w_hat)
    wx = to_real(grid, ddx(grid, w_hat))
    wy = to_real(grid, ddy(grid, w_hat))
    return -to_hat(grid, u * wx + v * wy)


def step(grid: Grid, w_hat, dt: float, tendency=None):
    """One Lawson IFRK4 step. ``tendency(w_hat) -> hat`` is an extra explicit term."""

    def rhs(wh):
        out = advection_hat(grid, wh) + grid.forcing_hat
        if tendency is not None:
            out = out + tendency(wh) * grid.mask
        return out

    e = jnp.exp(grid.linear * (0.5 * dt))
    e2 = e * e
    k1 = rhs(w_hat)
    k2 = rhs(e * (w_hat + 0.5 * dt * k1))
    k3 = rhs(e * w_hat + 0.5 * dt * k2)
    k4 = rhs(e2 * w_hat + dt * e * k3)
    out = e2 * w_hat + (dt / 6.0) * (e2 * k1 + 2.0 * e * (k2 + k3) + k4)
    return out if grid.exp_filter is None else out * grid.exp_filter


# ------------------------------------------------------------------ grids and filters
def truncate(fine: Grid, coarse: Grid, w_hat):
    """Sharp spectral filter from ``fine`` to ``coarse``: keep the coarse band (then the
    coarse grid's exponential filter, if it has one).

    rfft2 is unnormalised, so amplitudes scale by (n_c / n_f)^2.
    """
    nf, nc, kc = fine.n, coarse.n, coarse.kmax
    scale = (nc / nf) ** 2
    lead = w_hat.shape[:-2]
    out = jnp.zeros((*lead, nc, nc // 2 + 1), dtype=w_hat.dtype)
    out = out.at[..., : kc + 1, : kc + 1].set(w_hat[..., : kc + 1, : kc + 1] * scale)
    out = out.at[..., nc - kc :, : kc + 1].set(w_hat[..., nf - kc :, : kc + 1] * scale)
    out = out * coarse.mask
    return out if coarse.exp_filter is None else out * coarse.exp_filter


def pad(coarse: Grid, fine: Grid, w_hat):
    """Zero-pad a coarse spectrum onto a finer grid: the inverse of ``truncate`` on the coarse band."""
    nf, nc, kc = fine.n, coarse.n, coarse.kmax
    scale = (nf / nc) ** 2
    out = jnp.zeros((*w_hat.shape[:-2], nf, nf // 2 + 1), dtype=w_hat.dtype)
    out = out.at[..., : kc + 1, : kc + 1].set(w_hat[..., : kc + 1, : kc + 1] * scale)
    return out.at[..., nf - kc :, : kc + 1].set(w_hat[..., nc - kc :, : kc + 1] * scale)


def random_initial(grid: Grid, key, peak: float = 4.0, max_velocity: float = 3.0, batch=()):
    """A random vorticity field with an energy spectrum peaked at ``peak``."""
    shape = (*batch, grid.n, grid.n // 2 + 1)
    kr, ki = jax.random.split(key)
    phase = jax.random.normal(kr, shape) + 1j * jax.random.normal(ki, shape)
    k = jnp.sqrt(grid.k2)
    amp = k * jnp.exp(-((k - peak) ** 2) / (2.0 * 1.5**2))  # vorticity amplitude
    w_hat = phase * amp * grid.mask
    w_hat = w_hat.at[..., 0, 0].set(0.0)
    # independent draws at +kx and -kx in the ky = 0 column are not the rfft of a real
    # field; a round trip projects onto the real (Hermitian-consistent) part
    w_hat = to_hat(grid, to_real(grid, w_hat))
    u, v = velocity(grid, w_hat)
    vmax = jnp.max(jnp.sqrt(u**2 + v**2), axis=(-2, -1), keepdims=True)
    return w_hat * (max_velocity / vmax)


# ------------------------------------------------------------------ diagnostics
def energy_spectrum(grid: Grid, w_hat, kbins: int | None = None):
    """Shell-summed kinetic energy E(k) per leading index.

    The 2/3-rule mask is square, so its corners reach |k| = sqrt(2) kmax; the default
    bins cover them (bincount silently drops values past ``length``).
    """
    kbins = kbins or int(np.ceil(np.sqrt(2.0) * grid.kmax))
    norm = (1.0 / grid.n**2) ** 2
    weight = jnp.where(grid.ky == 0, 1.0, 2.0)  # rfft half plane
    e = 0.5 * jnp.abs(w_hat) ** 2 * grid.inv_k2 * norm * weight
    shell = jnp.round(jnp.sqrt(grid.k2)).astype(jnp.int32)
    flat_e = e.reshape(*e.shape[:-2], -1)
    flat_s = jnp.broadcast_to(shell, e.shape[-2:]).reshape(-1)
    return jax.vmap(lambda row: jnp.bincount(flat_s, weights=row, length=kbins + 1))(
        flat_e.reshape(-1, flat_e.shape[-1])).reshape(*e.shape[:-2], kbins + 1)


def correlation(a, b):
    """Pearson correlation of two real fields over the last two axes."""
    a = a - a.mean(axis=(-2, -1), keepdims=True)
    b = b - b.mean(axis=(-2, -1), keepdims=True)
    return (a * b).sum(axis=(-2, -1)) / jnp.sqrt((a * a).sum(axis=(-2, -1)) * (b * b).sum(axis=(-2, -1)))


def max_speed(grid: Grid, w_hat):
    u, v = velocity(grid, w_hat)
    return jnp.max(jnp.sqrt(u**2 + v**2))
