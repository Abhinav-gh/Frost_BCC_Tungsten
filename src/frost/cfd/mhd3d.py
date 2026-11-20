"""Incompressible MHD turbulence on the 3D pseudo-spectral grid (workload W5), composed from
``frost.cfd.spectral3d``.

State ``(..., 6, n, n, n//2+1)``: velocity (components 0-2) and magnetic field (3-5) in Alfven
units (b = B / sqrt(mu0 rho)), both divergence-free:

    du/dt = P[u x w + j x b] + nu lap u + f_u,      w = curl u, j = curl b
    db/dt = curl(u x b) + eta lap b + f_b

P removes the gradient part (the total pressure). The magnetic Prandtl number is 1 (eta = nu), so
``Grid3.linear`` serves both fields. Forcing injects constant power ``grid.power`` into |k| <= k_force
of each field separately (``spectral3d.forcing`` applied to u and to b), which keeps the turbulence
stationary with a magnetic field of the same order as the flow rather than waiting on a dynamo. The
time step is ``spectral3d.step``'s Lawson IFRK4 with this right-hand side and the same per-step
exponential filter (``Grid3.filter_strength``).

Classical LES closures for MHD (``les_tendency``):
- ``smag``: Smagorinsky for the flow, nu_t = (c Delta)^2 |S|, and for the field, eta_t = (c Delta)^2 |j|
  (the current-based magnetic diffusivity);
- ``cl``: Chollet-Lesieur spectral eddy viscosity from the total (kinetic plus magnetic) energy at the
  cutoff, applied to both fields;
- ``mixed``: ``smag`` plus the nonlinear gradient model for both the turbulent stress and the
  electromotive force, tau_ij = cg Delta^2 (d_l u_i d_l u_j - d_l b_i d_l b_j) and
  E = cg Delta^2 (d_l u) x (d_l b) (Clark et al. 1979; the MHD form of Grete et al. 2016), cg = 1/12.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from frost.cfd import spectral3d as s3
from frost.cfd.spectral3d import Grid3


def fields(s_hat):
    return s_hat[..., :3, :, :, :], s_hat[..., 3:, :, :, :]


def join(u_hat, b_hat):
    return jnp.concatenate([u_hat, b_hat], axis=-4)


def _cross(a, b):
    return jnp.stack([a[..., 1, :, :, :] * b[..., 2, :, :, :] - a[..., 2, :, :, :] * b[..., 1, :, :, :],
                      a[..., 2, :, :, :] * b[..., 0, :, :, :] - a[..., 0, :, :, :] * b[..., 2, :, :, :],
                      a[..., 0, :, :, :] * b[..., 1, :, :, :] - a[..., 1, :, :, :] * b[..., 0, :, :, :]], axis=-4)


def project(grid: Grid3, f_hat):
    """Divergence-free part of both fields."""
    fu, fb = fields(f_hat)
    return join(s3.project(grid, fu), s3.project(grid, fb))


def nonlinear(grid: Grid3, s_hat):
    """(P[(u x w + j x b)^], curl[(u x b)^]), dealiased."""
    u_hat, b_hat = fields(s_hat)
    u, b = s3.to_real(grid, u_hat), s3.to_real(grid, b_hat)
    w, j = s3.to_real(grid, s3.curl(grid, u_hat)), s3.to_real(grid, s3.curl(grid, b_hat))
    n_u = s3.project(grid, s3.to_hat(grid, _cross(u, w) + _cross(j, b)))
    n_b = s3.curl(grid, s3.to_hat(grid, _cross(u, b)))
    return join(n_u, n_b)


def forcing(grid: Grid3, s_hat):
    """Constant power into |k| <= k_force: ``grid.power`` into the flow; ``grid.power_b`` into the field
    (< 0: the same as the flow). Forcing the field directly amplifies its large scales and, with the
    alignment of u and b it drives, leaves no steady state (energy +45% over 10 units at 256^3 with
    0.05 into each field); flow-only forcing (power_b = 0) sustains the field by dynamo action."""
    u_hat, b_hat = fields(s_hat)
    pb = grid.power if grid.power_b < 0 else grid.power_b
    f_b = jnp.zeros_like(b_hat) if pb == 0 else s3.forcing(grid, b_hat) * (pb / max(grid.power, 1e-30))
    return join(s3.forcing(grid, u_hat), f_b)


def step(grid: Grid3, s_hat, dt: float, tendency=None):
    """One Lawson IFRK4 step; ``tendency(s_hat) -> hat`` is an extra explicit term (projected)."""

    def rhs(sh):
        out = nonlinear(grid, sh) + forcing(grid, sh)
        if tendency is not None:
            out = out + project(grid, tendency(sh) * grid.mask)
        return out

    e = jnp.exp(grid.linear * (0.5 * dt))
    e2 = e * e
    k1 = rhs(s_hat)
    k2 = rhs(e * (s_hat + 0.5 * dt * k1))
    k3 = rhs(e * s_hat + 0.5 * dt * k2)
    k4 = rhs(e2 * s_hat + dt * e * k3)
    out = e2 * s_hat + (dt / 6.0) * (e2 * k1 + 2.0 * e * (k2 + k3) + k4)
    return out if grid.exp_filter is None else out * grid.exp_filter


# ------------------------------------------------------------------ classical LES closures
def _grads(grid: Grid3, f_hat):
    kvec = s3._k(grid)
    return s3.to_real(grid, 1j * jnp.stack([f_hat * kj for kj in kvec], axis=-4))  # (..., 3 i, 3 j, ...): d_j f_i


def _div_stress(grid: Grid3, stress):
    """sum_j d_j T_ji for a symmetric real stress (..., 3, 3, n, n, n)."""
    kvec = s3._k(grid)
    tau = s3.to_hat(grid, stress)
    return sum(1j * kvec[j] * tau[..., j, :, :, :, :] for j in range(3))


def smagorinsky(grid: Grid3, cs: float = 0.1, cg: float = 0.0):
    """Smagorinsky for u (nu_t from |S|) and b (eta_t from |j|), plus, if cg > 0, the gradient model
    for the stress and the electromotive force. Delta = pi / k_c as in ``spectral3d.smagorinsky``."""
    delta2 = (np.pi / grid.les_cutoff) ** 2

    def tendency(s_hat):
        u_hat, b_hat = fields(s_hat)
        gu = _grads(grid, u_hat)
        s_ij = 0.5 * (gu + jnp.swapaxes(gu, -4, -5))
        s_norm = jnp.sqrt(2.0 * jnp.sum(s_ij**2, axis=(-5, -4)))
        stress = 2.0 * cs**2 * delta2 * s_norm[..., None, None, :, :, :] * s_ij
        j = s3.to_real(grid, s3.curl(grid, b_hat))
        eta_t = cs**2 * delta2 * jnp.sqrt(jnp.sum(j**2, axis=-4))
        emf = -eta_t[..., None, :, :, :] * j  # d b / dt += curl(-eta_t j) = magnetic diffusion
        if cg:
            gb = _grads(grid, b_hat)
            stress = stress - cg * delta2 * (jnp.einsum("...ikxyz,...jkxyz->...ijxyz", gu, gu)
                                             - jnp.einsum("...ikxyz,...jkxyz->...ijxyz", gb, gb))
            # E_i = cg Delta^2 eps_ijk d_l u_j d_l b_k, summed over l
            emf = emf + cg * delta2 * sum(_cross(gu[..., :, l, :, :, :], gb[..., :, l, :, :, :]) for l in range(3))
        return join(_div_stress(grid, stress), s3.curl(grid, s3.to_hat(grid, emf)))

    return tendency


def chollet_lesieur(grid: Grid3, c: float = 1.0, ck: float = 1.4):
    """``spectral3d.chollet_lesieur``'s kernel with nu_t from the total (kinetic plus magnetic) energy in
    the cutoff shell, applied to both fields (eta_t = nu_t)."""
    kc = grid.les_cutoff
    k = np.sqrt(grid.k2)
    cusp = np.where(k > 0, 1.0 + 34.5 * np.exp(-3.03 * kc / np.maximum(k, 1e-6)), 0.0)
    kernel = (cusp * grid.k2).astype(np.float32)
    shell = ((k >= kc - 0.5) & (k < kc + 0.5)).astype(np.float32)
    coef = c * 0.441 * ck**-1.5

    def tendency(s_hat):
        nu = coef * jnp.sqrt(jnp.maximum(s3.energy_of(grid, s_hat, shell), 0.0) / kc)  # sums all six components
        return -nu[..., None, None, None, None] * kernel * s_hat

    return tendency


def les_tendency(grid: Grid3, kind: str, c: float):
    """``none`` | ``smag`` (c = c_s) | ``cl`` (c scales nu_t) | ``mixed`` (smag with c_s = c plus the
    gradient model for stress and EMF)."""
    if kind == "none":
        return None
    if kind == "smag":
        return smagorinsky(grid, c)
    if kind == "mixed":
        return smagorinsky(grid, c, 1.0 / 12.0)
    if kind == "cl":
        return chollet_lesieur(grid, c)
    raise ValueError(kind)


# ------------------------------------------------------------------ initial states and diagnostics
def random_initial(grid: Grid3, key, peak: float = 3.0, rms: float = 0.5, batch=()):
    ku, kb = jax.random.split(key)
    return join(s3.random_initial(grid, ku, peak, rms, batch), s3.random_initial(grid, kb, peak, rms, batch))


def energies(grid: Grid3, s_hat):
    """(kinetic, magnetic) energy per leading index."""
    u_hat, b_hat = fields(s_hat)
    return s3.energy_of(grid, u_hat), s3.energy_of(grid, b_hat)


def cross_helicity(grid: Grid3, s_hat):
    """<u . b> (Parseval)."""
    u_hat, b_hat = fields(s_hat)
    return jnp.sum(jnp.real(jnp.conj(u_hat) * b_hat) * grid.weight, axis=(-4, -3, -2, -1)) / float(grid.n) ** 6


def dissipation(grid: Grid3, s_hat):
    """nu <|w|^2> + eta <|j|^2> (eta = nu)."""
    u_hat, b_hat = fields(s_hat)
    return 2.0 * grid.nu * (s3.energy_of(grid, s3.curl(grid, u_hat)) + s3.energy_of(grid, s3.curl(grid, b_hat)))


def max_signal_speed(grid: Grid3, s_hat):
    """max |u| + max |b|: the CFL speed (flow plus Alfven)."""
    u_hat, b_hat = fields(s_hat)
    return s3.max_speed(grid, u_hat) + s3.max_speed(grid, b_hat)
