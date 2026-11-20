"""Interface pinning: a solid-liquid slab held at a fixed solid fraction by a harmonic bias on the crystal's long-range
order, and the melting point read from where the bias's mean force vanishes (further_expansions_beyond.md 7.2).

Pedersen (J. Chem. Phys. 139, 104102, 2013; with Hummel, Kresse, Kahl and Dellago, Phys. Rev. B 88, 094101, 2013):

    Q = (1/K) sum_k |rho_k| / N,   rho_k = sum_j exp(i k . r_j),        k over K in-plane reciprocal vectors
    U = kappa / 2 (Q - a)^2,       F_j = -kappa (Q - a) dQ/dr_j,
    dQ/dr_j = 1 / (K N) sum_k k (S_k cos(k . r_j) - C_k sin(k . r_j)) / |rho_k|,   C_k + i S_k = rho_k.

Two-phase coexistence makes the Gibbs energy linear in Q between the liquid's Q_l and the solid's Q_s,
G(Q) = G_l + N dmu (Q - Q_l) / (Q_s - Q_l) with dmu = mu_s - mu_l per atom, so under the bias <Q> settles where the bias's
mean force balances the drive: dmu = kappa (Q_s - Q_l) (a - <Q>) / N. The melting point is where <Q> = a (dmu = 0); it
needs neither Q_s nor Q_l, which only set dmu's scale.

Wave vectors in the slab's plane (k_z = 0) make Q independent of L_z: under NP_zAT (``StochasticCellRescaling(axes="z")``)
the bias does no work on the barostat and adds nothing to P_zz. In Frost the bias is added to both force callables, the
teacher's and the student's, so the held residual F_T - F_S is unchanged and the bias is exact on every inner step, like
the analytic ZBL term of the cascade runs.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def inplane_kvectors(cell: np.ndarray, cells_xy: tuple[int, int], hkl=((2, 0, 0), (0, 2, 0))) -> np.ndarray:
    """Reciprocal vectors (A^-1) of a cubic crystal of cells_xy conventional cells across an orthorhombic box: (h, k, 0) ->
    2 pi (h nx / Lx, k ny / Ly, 0). The fcc (200) and (020) reflections are the strongest in-plane ones (all four atoms of
    the conventional cell in phase)."""
    lx, ly = float(cell[0, 0]), float(cell[1, 1])
    nx, ny = cells_xy
    out = []
    for h, k, l_ in hkl:
        if l_:
            raise ValueError("in-plane vectors only (l = 0): Q must not depend on L_z")
        out.append((2 * math.pi * h * nx / lx, 2 * math.pi * k * ny / ly, 0.0))
    return np.asarray(out, dtype=np.float64)


class Pinning:
    """The bias U = kappa/2 (Q - a)^2 on Q (above); kappa in eV, a dimensionless."""

    def __init__(self, kvecs, kappa: float, a: float, n_atoms: int, device="cpu"):
        self.k = torch.as_tensor(np.asarray(kvecs), dtype=torch.float64, device=device)  # (K, 3)
        self.kappa, self.a, self.n = float(kappa), float(a), int(n_atoms)

    def _sums(self, x: torch.Tensor):
        phase = x.to(torch.float64) @ self.k.T  # (N, K)
        cos, sin = torch.cos(phase), torch.sin(phase)
        c, s = cos.sum(0), sin.sum(0)
        return cos, sin, c, s, torch.sqrt(c * c + s * s)

    def q(self, x: torch.Tensor) -> torch.Tensor:
        """Q as a 0-d tensor on x's device (no host sync)."""
        *_, amp = self._sums(x)
        return amp.mean() / self.n

    def __call__(self, x: torch.Tensor):
        """(bias energy, bias forces (N, 3) float64, Q), all tensors on x's device."""
        cos, sin, c, s, amp = self._sums(x)
        q = amp.mean() / self.n
        # dQ/dr_j = sum_k k (S_k cos_jk - C_k sin_jk) / |rho_k| / (K N)
        w = (s.unsqueeze(0) * cos - c.unsqueeze(0) * sin) / amp.clamp_min(1e-12).unsqueeze(0)  # (N, K)
        dq = (w @ self.k) / (self.k.shape[0] * self.n)
        dev = q - self.a
        return 0.5 * self.kappa * dev * dev, -self.kappa * dev * dq, q


def delta_mu(kappa: float, a: float, q_mean: float, q_solid: float, q_liquid: float, n_atoms: int) -> float:
    """mu_solid - mu_liquid per atom (eV) from the pinned mean of Q (Pedersen 2013)."""
    return kappa * (q_solid - q_liquid) * (a - q_mean) / n_atoms


def block_se(series, n_blocks: int = 10) -> float:
    """Standard error of a correlated series' mean from n_blocks contiguous block means (Flyvbjerg and Petersen 1989:
    blocks longer than the correlation time make the block means independent)."""
    v = np.asarray(series, dtype=np.float64)
    if len(v) < 2 * n_blocks:
        return float("nan")
    m = len(v) // n_blocks
    means = v[: m * n_blocks].reshape(n_blocks, m).mean(1)
    return float(means.std(ddof=1) / math.sqrt(n_blocks))


def melting_point(temps, gaps, ses) -> tuple[float, float]:
    """T_m and its standard error from (a - <Q>) measured at several temperatures: a weighted straight line through
    gap(T) = s (T - T_m) (dmu is linear in T near T_m), crossing zero at T_m; the error by propagating the fit's covariance."""
    t, g, e = (np.asarray(v, dtype=np.float64) for v in (temps, gaps, ses))
    w = 1.0 / np.maximum(e, 1e-12) ** 2
    a_mat = np.stack([np.ones_like(t), t], 1)
    cov = np.linalg.inv(a_mat.T @ (w[:, None] * a_mat))
    b0, b1 = cov @ a_mat.T @ (w * g)
    tm = -b0 / b1
    grad = np.array([-1.0 / b1, b0 / b1 ** 2])  # d tm / d(b0, b1)
    return float(tm), float(math.sqrt(grad @ cov @ grad))
