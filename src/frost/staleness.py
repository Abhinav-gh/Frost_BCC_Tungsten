"""Staleness statistics for multirate schedules (pilot E4, student E5).

A multirate schedule refreshes an expensive quantity s_t (a residual force, a
closure flux) every k steps and holds or extrapolates it in between. These are
the statistics that decide how large k can be. They are computed over every
atom of every frame in a window, never over a single atom or edge. Arrays have
shape (windows, frames, atoms, 3) unless stated otherwise.

    drift(k)   RMS |s(t+k) - s(t)|
    hold(k)    RMS over j = 1..k of |s(t+j) - s(t)|: mean staleness over an
               outer step that holds s
    linear(k)  as hold, with s extrapolated linearly from the previous outer step
    lagged(k)  as linear, one outer step later: a pipeline whose expensive call
               lags by one outer step
"""

from __future__ import annotations

import math

import numpy as np

KB_EV = 8.617333262e-5
SIGMA_ACC_0P6 = 1.0511  # erfc(sigma / 2 sqrt 2) = 0.6


def rms(x: np.ndarray) -> float:
    """RMS of per-atom vector norms; the last axis holds the 3 components."""
    return float(np.sqrt((x**2).sum(-1).mean()))


def per_mass(s: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """A residual divided by mass: the acceleration error the integrator actually follows.

    ``s`` has shape (..., atoms, 3) and ``masses`` (atoms,) in amu, so a force
    residual in eV/A becomes eV/(A amu). The same force error on a 1 amu hydrogen
    is 16-28x the acceleration error it causes on oxygen or silicon, which is why
    a force budget is not comparable across systems (plan section 1.11).
    """
    m = np.asarray(masses, dtype=float).reshape(*(1,) * (s.ndim - 2), -1, 1)
    return s / m


def drift(s: np.ndarray, k: int) -> float:
    """Every pair (t, t+k) inside a window."""
    return rms(s[:, k:] - s[:, :-k])


def hold(s: np.ndarray, k: int) -> float:
    acc = [((s[:, j:] - s[:, :-j]) ** 2).sum(-1).mean() for j in range(1, k + 1)]
    return float(np.sqrt(np.mean(acc)))


def linear(s: np.ndarray, k: int) -> float | None:
    n = s.shape[1]
    if 2 * k > n - 1:
        return None
    acc = []
    for j in range(1, k + 1):
        t = np.arange(k, n - j)
        pred = s[:, t] + (j / k) * (s[:, t] - s[:, t - k])
        acc.append(((s[:, t + j] - pred) ** 2).sum(-1).mean())
    return float(np.sqrt(np.mean(acc)))


def lagged(s: np.ndarray, k: int) -> float | None:
    """Extrapolate from s(t-k), s(t) to s(t+k+j), j = 1..k: one outer step of lag."""
    n = s.shape[1]
    if 3 * k > n - 1:
        return None
    acc = []
    for j in range(1, k + 1):
        t = np.arange(k, n - k - j)
        pred = s[:, t] + ((k + j) / k) * (s[:, t] - s[:, t - k])
        acc.append(((s[:, t + k + j] - pred) ** 2).sum(-1).mean())
    return float(np.sqrt(np.mean(acc)))


def energy_stats(e: np.ndarray, n_d: int, temperature: float, n_atoms: int) -> dict:
    """Spread of an energy residual's change over n_d steps; e has shape (windows, frames).

    The acceptance proxy erfc(sigma / 2 sqrt 2), sigma = sd / kT, is what a
    Metropolis test on that change accepts when it is Gaussian (note section
    5.3); the block size is the number of atoms that keeps it at 0.6, assuming
    the variance is extensive.
    """
    d = (e[:, n_d:] - e[:, :-n_d]).ravel()
    sd = float(d.std())
    sigma = sd / (KB_EV * temperature)
    return {
        "sd_eV": sd,
        "sd_per_sqrtN_meV": 1e3 * sd / math.sqrt(n_atoms),
        "sigma_kT": sigma,
        "acceptance_proxy": math.erfc(sigma / (2 * math.sqrt(2))),
        "n_atoms_for_acceptance_0p6": n_atoms * (SIGMA_ACC_0P6 / sigma) ** 2 if sigma > 0 else None,
    }
