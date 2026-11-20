"""The ZBL universal screened nuclear repulsion with a smooth switch, for radiation-damage cascades
(further_expansions_beyond.md 6.9).

A primary knock-on atom drives atoms closer than any learned potential was trained on (MACE-MP-0 sees little below
~1.5 A); production cascade codes blend in the universal Ziegler-Biersack-Littmark repulsion there (Byggmastar et al.
2019; NEP-ZBL, Liu et al. 2023). In Frost it belongs to the fast force: an analytic pair term, cheap and exact on every
inner step, never held; only the smooth many-body part (teacher minus student) is held across an interval.

    V(r) = 14.399645 Z1 Z2 / r * phi(r / a) * S(r)     (eV, A)
    phi(x) = 0.18175 e^-3.19980x + 0.50986 e^-0.94229x + 0.28022 e^-0.40290x + 0.02817 e^-0.20162x
    a = 0.46850 / (Z1^0.23 + Z2^0.23)
S(r) is 1 below r_in and falls smoothly (quintic, zero slope and curvature at both ends) to 0 at r_out.
"""

from __future__ import annotations

import torch

COULOMB = 14.399645  # e^2 / (4 pi eps0), eV A
PHI_C = (0.18175, 0.50986, 0.28022, 0.02817)
PHI_D = (3.19980, 0.94229, 0.40290, 0.20162)


def switch(r: torch.Tensor, r_in: float, r_out: float) -> torch.Tensor:
    t = ((r - r_in) / (r_out - r_in)).clamp(0.0, 1.0)
    return 1.0 - t ** 3 * (10.0 - 15.0 * t + 6.0 * t * t)


def zbl_pair(r: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    a = 0.46850 / (z1 ** 0.23 + z2 ** 0.23)
    x = r / a
    phi = sum(c * torch.exp(-d * x) for c, d in zip(PHI_C, PHI_D, strict=True))
    return COULOMB * z1 * z2 / r * phi


class ZBL:
    """Energy and forces of the switched ZBL repulsion over a full (i, j) edge list with shift vectors."""

    def __init__(self, numbers: torch.Tensor, r_in: float = 1.0, r_out: float = 2.0):
        self.z = numbers.to(torch.float64)
        self.r_in, self.r_out = r_in, r_out

    def __call__(self, x: torch.Tensor, i: torch.Tensor, j: torch.Tensor, shifts: torch.Tensor):
        x = x.detach().to(torch.float64).requires_grad_(True)
        d = x[j] - x[i] + shifts.to(torch.float64)
        r = d.norm(dim=1)
        keep = r < self.r_out
        rk = r[keep]
        e_pair = zbl_pair(rk, self.z[i[keep]], self.z[j[keep]]) * switch(rk, self.r_in, self.r_out)
        energy = 0.5 * e_pair.sum()  # a full list counts every pair twice
        (grad,) = torch.autograd.grad(energy, x)
        return energy.detach(), -grad
