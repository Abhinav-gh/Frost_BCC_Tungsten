"""Neighbour lists for the multirate loop.

Convention, shared by vesin, matscipy and MACE: edge (i, j) with integer image
S has vector r_ij = x_j - x_i + S @ cell. MACE calls i the sender and j the
receiver; the student sums over edges leaving i.

A Verlet list built at a model's cutoff plus a skin serves that model between
rebuilds, because each model masks edges beyond its own cutoff. The online loop
keeps one list per model. The student's skin is tight, because its call cost
grows with the number of edges it carries.

Lists can be built from unwrapped positions. The image offsets then absorb the
wrapping, positions never jump, and each list rebuilds on its own schedule while
staying consistent with the others. Three builders produce the same list: vesin on
the CPU, NVIDIA ALCHEMI Toolkit-Ops' cell list on the GPU, and an all-pairs search
under the minimum image (``edges_brute``) for small orthorhombic cells; "auto" picks
the all-pairs search where it is valid and small, the cell list otherwise.
"""

from __future__ import annotations

import math

import numpy as np
import torch

BUILDERS = ("vesin", "alchemi", "brute", "auto")
BRUTE_MAX_ATOMS = 3000  # "auto" uses the all-pairs search up to here: N^2 pair vectors, ~0.2 GB at 3,000 atoms


def wrap(x: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Positions mapped into the cell. Forces and energies are unchanged."""
    frac = x @ np.linalg.inv(cell)
    return (frac - np.floor(frac)) @ cell


def edges_for(x, cell, cutoff: float, device="cpu", dtype=torch.float32,
              wrap_positions: bool = True):
    """Full periodic edge list, built on the CPU with vesin.

    Returns (x_out, i, j, shifts, unit_shifts) with r_ij = x_out[j] - x_out[i] + shifts.
    With ``wrap_positions`` x_out is x wrapped into the cell, and vectors must be
    computed from x_out, not the input. Without it x_out is x itself.
    """
    from vesin import NeighborList

    cell = np.asarray(cell, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    frac = x @ np.linalg.inv(cell)
    image = np.floor(frac)
    xw = (frac - image) @ cell
    i, j, S = NeighborList(cutoff=float(cutoff), full_list=True).compute(
        points=np.ascontiguousarray(xw),
        box=np.ascontiguousarray(cell),
        periodic=True,
        quantities="ijS",
    )
    i = np.asarray(i, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    S = np.asarray(S, dtype=np.int64)
    if not wrap_positions:
        # x = xw + image @ cell, so these offsets give the same vectors from x
        S = S + (image[i] - image[j]).astype(np.int64)
    return (
        torch.as_tensor(xw if wrap_positions else x, dtype=dtype, device=device),
        torch.as_tensor(i, device=device),
        torch.as_tensor(j, device=device),
        torch.as_tensor(S.astype(np.float64) @ cell, dtype=dtype, device=device),
        torch.as_tensor(S, device=device),
    )


def edges_on_gpu(x: torch.Tensor, cell, cutoff: float, wrap_positions: bool = True,
                 max_neighbours: int | None = None):
    """``edges_for`` built on the GPU with NVIDIA ALCHEMI Toolkit-Ops' cell list.

    Positions are wrapped before the search, so the result does not depend on how
    the kernel treats unwrapped input. The per-atom neighbour capacity starts at
    twice the mean count at this density and doubles whenever the kernel reports
    an overflow. Returns tensors on ``x``'s device and dtype.
    """
    from nvalchemiops.neighbors.neighbor_utils import NeighborOverflowError
    from nvalchemiops.torch.neighbors import neighbor_list

    cell64 = np.asarray(cell, dtype=np.float64)
    cell_t = torch.as_tensor(cell64, dtype=x.dtype, device=x.device)
    image = torch.floor(x @ torch.linalg.inv(cell_t))
    xw = x - image @ cell_t
    pbc = torch.ones(3, dtype=torch.bool, device=x.device)
    if max_neighbours is None:
        density = x.shape[0] / abs(float(np.linalg.det(cell64)))
        max_neighbours = int(2.0 * density * 4.0 / 3.0 * math.pi * cutoff**3) + 16
    while True:
        try:
            pairs, _, unit = neighbor_list(xw, float(cutoff), cell=cell_t, pbc=pbc,
                                           return_neighbor_list=True, method="cell_list",
                                           max_neighbors=max_neighbours)
            break
        except NeighborOverflowError:
            max_neighbours *= 2
    i, j = pairs[0].long(), pairs[1].long()
    unit = unit.long()
    if not wrap_positions:
        unit = unit + (image[i] - image[j]).long()
    return (xw if wrap_positions else x), i, j, unit.to(x.dtype) @ cell_t, unit


def brute_ok(cell, radius: float, n_atoms: int | None = None) -> bool:
    """Whether ``edges_brute`` applies: an orthorhombic cell whose every length exceeds twice the radius (then each pair
    has at most one image within it, the nearest), and, if given, at most BRUTE_MAX_ATOMS atoms."""
    c = np.asarray(cell, dtype=np.float64)
    if not np.allclose(c, np.diag(np.diag(c)), atol=1e-9):
        return False
    return bool(np.all(np.diag(c) > 2.0 * radius)) and (n_atoms is None or n_atoms <= BRUTE_MAX_ATOMS)


def edges_brute(x: torch.Tensor, cell, cutoff: float, wrap_positions: bool = True):
    """``edges_for`` by testing all pairs under the minimum image, on x's device (2026-11-13).

    For a small cell a cell-list build is dominated by fixed costs: 2.6 ms for 648 waters, rebuilt on 127 of 220 steps
    by the student's tight list (fast hydrogens), which made it most of the cheap student's inner step (gaps.md, water
    profile). Here it is a handful of kernels on N^2 pairs and one size read. Needs ``brute_ok(cell, cutoff)``.
    """
    cell64 = np.asarray(cell, dtype=np.float64)
    if not brute_ok(cell64, cutoff):
        raise ValueError("edges_brute needs an orthorhombic cell longer than twice the cutoff on every axis")
    lengths = torch.as_tensor(np.diag(cell64), dtype=x.dtype, device=x.device)
    image = torch.floor(x / lengths)
    fw = x / lengths - image  # wrapped fractional coordinates
    d = fw.unsqueeze(0) - fw.unsqueeze(1)  # (i, j): frac_j - frac_i
    near = -torch.round(d)  # the image of j nearest to i
    r = (d + near) * lengths
    mask = (r * r).sum(-1) < cutoff * cutoff
    mask.fill_diagonal_(False)
    i, j = mask.nonzero(as_tuple=True)
    unit = near[i, j].long()
    if not wrap_positions:  # vectors from the unwrapped x: x = xw + image @ cell
        unit = unit + (image[i] - image[j]).long()
    cell_t = torch.as_tensor(cell64, dtype=x.dtype, device=x.device)
    xw = fw * lengths
    return (xw if wrap_positions else x), i, j, unit.to(x.dtype) @ cell_t, unit


class VerletList:
    """Edges within cutoff + skin, rebuilt once any atom has moved half the skin.

    With ``wrap_positions`` a rebuild wraps positions into the cell, and the caller
    must continue from the returned positions so vectors and shifts stay
    consistent. Without it positions come back unchanged, and several lists can
    rebuild independently on the same trajectory. ``builder`` is "vesin" (CPU) or
    "alchemi" (GPU); both give the same edges.
    """

    def __init__(self, cell, cutoff: float, skin: float = 1.0, device="cuda",
                 wrap_positions: bool = True, builder: str = "vesin",
                 outer: "VerletList | None" = None, lag_margin: float | None = None):
        """``outer``: a list of larger radius on the same unwrapped trajectory. A rebuild
        then filters the outer list's edges instead of building a cell list: once the
        outer list is valid at x it holds every pair within its own cutoff, so it holds
        every pair this list needs whenever cutoff + skin <= outer.cutoff (a Verlet list
        of a Verlet list). On liquid Cu a cell-list build cost 3.9 ms, every ~5 steps of
        the student's tight skin (2026-10-02)."""
        if builder not in BUILDERS:
            raise ValueError(f"unknown neighbour-list builder {builder!r}; choose from {BUILDERS}")
        self.cell = np.asarray(cell, dtype=np.float64)
        self.cutoff = float(cutoff)
        self.skin = float(skin)
        self.device = device
        self.wrap_positions = wrap_positions
        self.builder = builder
        self.rebuilds = 0
        self.x_ref: torch.Tensor | None = None
        self.i = self.j = self.shifts = self.unit_shifts = None
        self.outer = outer
        # lag_margin: decide each step from the *previous* step's displacement, copied to the
        # host without blocking, against half the skin less the margin. No per-step host sync,
        # so the GPU never idles while the host waits. Only for a list consulted every step: a
        # pair that slips one step past the list boundary enters MACE's p = 6 envelope (a triple
        # zero at r_cut) with weight ~56 eps^3, ~1e-4 of a full pair at Cu's 2 fs step
        self.lag_margin = lag_margin
        self._pinned = None
        self._event = None
        if outer is not None:
            if wrap_positions or outer.wrap_positions:
                raise ValueError("a derived list needs unwrapped positions on both lists")
            if cutoff + skin > outer.cutoff:
                raise ValueError(f"cutoff + skin {cutoff + skin} exceeds the outer cutoff {outer.cutoff}")

    def set_cell(self, cell) -> None:
        """The cell changed (a barostat moved it): the same edges, new shift vectors (shifts = unit_shifts @ cell,
        checked 2026-11-13); the next rebuild searches in the new cell. Positions must be scaled by the caller, and the
        displacement test then counts the affine part of the move too, which only makes rebuilds earlier (safe)."""
        self.cell = np.asarray(cell, dtype=np.float64)
        if self.unit_shifts is not None:
            c = torch.as_tensor(self.cell, dtype=torch.float64, device=self.unit_shifts.device)
            self.shifts = (self.unit_shifts.to(torch.float64) @ c).to(self.shifts.dtype)

    def _still_valid_lagged(self, x: torch.Tensor) -> bool:
        moved = (x - self.x_ref).norm(dim=1).max().reshape(1)
        previous = None
        if self._event is not None:
            self._event.synchronize()  # the previous step's copy: done long ago, no stall
            previous = float(self._pinned[0])
        if self._pinned is None or self._pinned.dtype != moved.dtype:
            self._pinned = torch.zeros(1, dtype=moved.dtype, pin_memory=True)
        self._pinned.copy_(moved, non_blocking=True)
        self._event = torch.cuda.Event()
        self._event.record()
        return previous is None or previous <= 0.5 * self.skin - self.lag_margin

    def update(self, x: torch.Tensor, force: bool = False, margin: float = 0.0) -> tuple[torch.Tensor, bool]:
        """Rebuild when an atom has moved more than half the skin less ``margin`` (A).

        ``margin`` > 0 rebuilds ahead of need: a caller about to run many steps without checking (a CUDA-graph
        chunk) passes the distance an atom can cover in them, so the list cannot run out mid-chunk.
        """
        if self.x_ref is not None and not force:
            if self.lag_margin is not None and x.is_cuda and margin <= 0.0:
                if self._still_valid_lagged(x):
                    return x, False
            else:
                moved = float((x - self.x_ref).norm(dim=1).max())
                if moved <= 0.5 * self.skin - margin:
                    return x, False
        self._event = None  # a new reference: the next lagged check starts afresh
        radius = self.cutoff + self.skin
        if self.outer is not None:
            self.outer.update(x)  # valid at x: rebuilds only if it, too, has run out of skin
            o = self.outer
            r = x[o.j] - x[o.i] + o.shifts.to(x.dtype)
            keep = (r.norm(dim=1) < radius).nonzero().squeeze(1)
            self.i, self.j = o.i[keep], o.j[keep]
            self.shifts, self.unit_shifts = o.shifts[keep], o.unit_shifts[keep]
            self.x_ref = x.detach().clone()
            self.rebuilds += 1
            return x, True
        builder = self.builder
        if builder == "auto":
            builder = "brute" if brute_ok(self.cell, radius, x.shape[0]) else ("alchemi" if x.is_cuda else "vesin")
        if builder == "brute":
            xo, i, j, shifts, unit = edges_brute(x.detach(), self.cell, radius, self.wrap_positions)
        elif builder == "alchemi":
            xo, i, j, shifts, unit = edges_on_gpu(x.detach(), self.cell, radius, self.wrap_positions)
        else:
            xo, i, j, shifts, unit = edges_for(x.detach().cpu().numpy(), self.cell, radius,
                                               self.device, x.dtype, self.wrap_positions)
        if not self.wrap_positions:
            xo = x
        self.i, self.j, self.shifts, self.unit_shifts = i, j, shifts, unit
        self.x_ref = xo.detach().clone()
        self.rebuilds += 1
        return xo, True


def tight_edges(lst, x, r_cut2: float):
    """The edges of a Verlet list (built at r_max + skin) that are within r_max at positions x: (i, j, shifts,
    unit_shifts). A distance check and a compaction, for a model call that should not evaluate the skin's pairs: MACE
    computes every edge it is given and zeroes the skin's only through its envelope (2026-11-16, teacher_list_cost.py:
    the call 21-34% cheaper on five systems, forces identical)."""
    r = x[lst.j].double() - x[lst.i].double() + lst.shifts.double()
    keep = ((r * r).sum(1) < r_cut2).nonzero(as_tuple=True)[0]
    return lst.i[keep], lst.j[keep], lst.shifts[keep], lst.unit_shifts[keep]


def concat_lists(lists, n_atoms: int):
    """B copies' Verlet lists as one disconnected graph over B x N atoms, copy b's atoms offset by b x N: (i, j, shifts,
    unit_shifts). Path-integral beads call a model once on every copy (experiments/e15_pimd/pimd_runtime.py)."""
    i = torch.cat([lst.i + b * n_atoms for b, lst in enumerate(lists)])
    j = torch.cat([lst.j + b * n_atoms for b, lst in enumerate(lists)])
    return i, j, torch.cat([lst.shifts for lst in lists]), torch.cat([lst.unit_shifts for lst in lists])


def filter_within(i, j, shifts, unit_shifts, x, r_cut2: float):
    """``tight_edges`` on edge arrays: the edges within sqrt(r_cut2) at positions x. For a list concatenated over copies
    (``concat_lists``, x over B x N atoms) it filters every copy at its own positions with one compaction, one host sync
    per call whatever B is (2026-11-17)."""
    r = x[j].double() - x[i].double() + shifts.double()
    keep = ((r * r).sum(1) < r_cut2).nonzero(as_tuple=True)[0]
    return i[keep], j[keep], shifts[keep], unit_shifts[keep]
