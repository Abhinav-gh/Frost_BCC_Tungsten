"""Local staleness: the magnitude clause measured region by region (2026-11-14).

The magnitude clause bounds the RMS change of the held acceleration over all atoms. When the fast physics is local (a
cascade's core, a defect, a reaction, an interface) a few hundred atoms can carry many times the budget while the RMS over
the whole cell stays under it: 100 hot atoms among 8,192 can be 9x over before the whole-cell RMS reaches the budget.

These helpers split the box into n x n x n regions (by the current positions, wrapped) and report the per-component RMS
in every region, and the bound each region is held to: epsilon x (1 + z / sqrt(6 m)) for a region of m atoms. The RMS of
3m independent Gaussian components has relative standard deviation 1 / sqrt(6 m), so with z = 4 a homogeneous state (the
whole-cell clause's case) crosses the bound in any of 64 regions with probability ~0.2% per check: the local clause stays
silent wherever staleness is spread evenly and binds only where it concentrates. Regions with fewer than ``min_atoms``
atoms are not judged (their scatter is too large to bound).
"""

from __future__ import annotations

import math

import torch


def region_index(x: torch.Tensor, box: torch.Tensor, n: int) -> torch.Tensor:
    """(N,) region ids in [0, n^3) of positions x (N, 3), unwrapped or not. ``box`` is the three lengths of an
    orthorhombic box, or a 3 x 3 cell whose rows are the lattice vectors (regions are then slabs of fractional space)."""
    f = torch.remainder(x @ torch.linalg.inv(box) if box.dim() == 2 else x / box, 1.0)
    ijk = torch.clamp((f * n).long(), min=0, max=n - 1)
    return (ijk[:, 0] * n + ijk[:, 1]) * n + ijk[:, 2]


def region_factor(m, z: float = 4.0):
    """The bound's allowance for a region of m atoms: 1 + z / sqrt(6 m) (tensor or number)."""
    if isinstance(m, torch.Tensor):
        return 1.0 + z / torch.sqrt(6.0 * m.clamp(min=1.0))
    return 1.0 + z / math.sqrt(6.0 * max(float(m), 1.0))


def staleness_stats(da: torch.Tensor, x: torch.Tensor, box: torch.Tensor, n: int = 4, z: float = 4.0,
                    min_atoms: int = 16) -> dict:
    """Whole-cell, worst-region and worst-atom staleness of a change of acceleration ``da`` (N, 3), any units.

    Returns per-component RMS values in da's units: ``rms`` (all atoms, the magnitude clause's number), ``local`` (the
    worst region's RMS divided by its allowance, so the local clause fails at ``local > epsilon``, like ``rms``),
    ``region`` (that region's RMS itself), ``atoms`` (atoms in it) and ``atom`` (the worst single atom's |da| /
    sqrt(3)). One host synchronisation."""
    sq = (da * da).sum(1)
    idx = region_index(x, box, n)
    tot = torch.zeros(n ** 3, dtype=sq.dtype, device=sq.device).index_add_(0, idx, sq)
    cnt = torch.zeros(n ** 3, dtype=sq.dtype, device=sq.device).index_add_(0, idx, torch.ones_like(sq))
    reg = torch.sqrt(tot / (3.0 * cnt.clamp(min=1.0)))
    scaled = torch.where(cnt >= min_atoms, reg / region_factor(cnt, z), torch.zeros_like(reg))
    w = torch.argmax(scaled)
    rms, local, region, atoms, atom = torch.stack([torch.sqrt(sq.mean() / 3.0), scaled[w], reg[w], cnt[w],
                                                   torch.sqrt(sq.max() / 3.0)]).tolist()
    return {"rms": rms, "local": local, "region": region, "atoms": int(atoms), "atom": atom}
