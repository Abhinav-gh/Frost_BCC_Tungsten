"""An online stiffness correction along bonds: the student's fast-mode error, fitted from the refreshes.

A held residual R = F_T - F_S stays fresh only if R varies slowly, and along a stiff bond it does not: a student
whose bond stiffness is off by dK leaves R ~ -dK dr e oscillating with the stretch. In water the certificate's
hydrogen budget needs the O-H stiffness right to ~0.35%, which no small student holds. Every refresh already
measures what is needed to fix it: the residual's change dR over the interval and the bond-length change dr. Their
least-squares slope along the bond is the remaining stiffness error, and adding the harmonic term

    E_b = 1/2 sum_b dK_kind(b) (r_b - r0_b)^2

to the student removes it, with no extra teacher call. The term is a potential, so the student stays
conservative; after each update the residual is recomputed from the same teacher call (CLAUDE.md trap 8).
"""

from __future__ import annotations

import torch


class BondStiffness:
    def __init__(self, bonds: torch.Tensor, kinds: torch.Tensor, x0: torch.Tensor, cell, n_kinds: int,
                 memory: float = 0.7, k_max: float = 50.0):
        self.a, self.b = bonds[:, 0].long(), bonds[:, 1].long()
        self.kinds = kinds.long()
        self.cell = torch.as_tensor(cell, dtype=x0.dtype, device=x0.device)
        self.inv = torch.linalg.inv(self.cell)
        self.r0 = self.geometry(x0)[0]
        self.dk = torch.zeros(n_kinds, dtype=x0.dtype, device=x0.device)
        self.memory, self.k_max = memory, k_max  # EWMA of the slope; |dK| bound, eV/A^2
        self.history: list[list[float]] = []

    @staticmethod
    def detect(numbers, x, cell, pairs=((1, 8),), cutoff: float = 1.3):
        """Bonds a-b for each (Z_a, Z_b) in ``pairs``: every Z_a atom to its nearest Z_b atom within ``cutoff``."""
        x = torch.as_tensor(x, dtype=torch.float64)
        cell_t = torch.as_tensor(cell, dtype=torch.float64)
        inv = torch.linalg.inv(cell_t)
        z = torch.as_tensor(numbers)
        bonds, kinds = [], []
        for kind, (za, zb) in enumerate(pairs):
            ia, ib = (z == za).nonzero().squeeze(1), (z == zb).nonzero().squeeze(1)
            d = x[ia][:, None, :] - x[ib][None, :, :]
            d = d - torch.round(d @ inv) @ cell_t
            dist = d.norm(dim=-1)
            near = dist.argmin(1)
            ok = dist[torch.arange(len(ia)), near] < cutoff
            bonds.append(torch.stack([ia[ok], ib[near[ok]]], 1))
            kinds.append(torch.full((int(ok.sum()),), kind))
        return torch.cat(bonds), torch.cat(kinds)

    def geometry(self, x: torch.Tensor):
        v = x[self.a] - x[self.b]
        v = v - torch.round(v @ self.inv) @ self.cell  # minimum image: a starting frame is wrapped into the cell
        r = v.norm(dim=1)
        return r, v / r.unsqueeze(1)

    def energy_forces(self, x: torch.Tensor):
        r, e = self.geometry(x)
        k = self.dk[self.kinds]
        stretch = r - self.r0
        energy = 0.5 * (k * stretch * stretch).sum()
        f = -(k * stretch).unsqueeze(1) * e  # on a; the opposite on b
        forces = torch.zeros_like(x).index_add_(0, self.a, f).index_add_(0, self.b, -f)
        return energy, forces

    def update(self, d_residual: torch.Tensor, x_old: torch.Tensor, x_new: torch.Tensor) -> None:
        """Fold one interval's residual change into dK (least squares along the bonds, smoothed)."""
        r_old, e_old = self.geometry(x_old)
        r_new, _ = self.geometry(x_new)
        dr = r_new - r_old
        proj = 0.5 * ((d_residual[self.a] - d_residual[self.b]) * e_old).sum(1)  # ~ -(remaining dK) dr
        n_k = self.dk.shape[0]
        num = torch.zeros(n_k, dtype=dr.dtype, device=dr.device).index_add_(0, self.kinds, proj * dr)
        den = torch.zeros(n_k, dtype=dr.dtype, device=dr.device).index_add_(0, self.kinds, dr * dr)
        # this interval's remaining error (measured with the current dK), so the estimate of the total
        # teacher-minus-student stiffness is dK - slope; smooth that total, not sums taken under older dK
        slope = torch.where(den > 0, num / den.clamp_min(1e-30), torch.zeros_like(num))
        self.dk = (self.dk - (1.0 - self.memory) * slope).clamp(-self.k_max, self.k_max)
        self.history.append(self.dk.tolist())
