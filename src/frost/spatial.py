"""Spatial multirate: the teacher's force refreshed often where motion is fast and rarely where it is slow
(further_expansions_beyond.md 7.3, tier 3).

An interface has two spectra: water's fastest period (~9 fs, ~15 fs with hydrogens at 4 amu) and a metal's (~110 fs for
Cu). One global interval must obey the fastest. Here the held residual R = F_T - F_S is split by atoms: the fast region A
(water and the metal's interface layers) is refreshed every k steps from a teacher call on a subsystem, A plus a buffer;
everything is refreshed every K = m k steps from the full teacher, which also replaces A's subsystem residual with the
exact one. The two residuals are applied as RESPA impulses on their own clocks (``SpatialSchedule``), each a kick to a
disjoint set of momenta, so the composition stays symplectic and time-reversible.

``RegionTeacher``: MACE's force on A from the subsystem C = A plus the atoms within ``buffer`` of A, summing only the site
energies of atoms within ``inner`` of A (MACE's public ``node_energy`` output, differentiated with autograd; no library
code touched). A site energy depends on atoms within L x r_c (L message-passing layers), so a force on an atom of A is exact
when every site energy that depends on it is complete inside C: buffer >= 2 L r_c (24 A for MACE-MP-0). A smaller buffer
is an approximation, and the runtime measures it exactly at every full refresh (the subsystem residual against the full
one at the same positions), which the certificate bounds like any other staleness.
"""

from __future__ import annotations

import numpy as np
import torch


def atoms_within(x: np.ndarray, cell: np.ndarray, centre_idx: np.ndarray, radius: float) -> np.ndarray:
    """Indices of atoms within ``radius`` (minimum image, orthorhombic or general cell) of any atom in ``centre_idx``."""
    inv = np.linalg.inv(cell)
    near = np.zeros(len(x), dtype=bool)
    near[centre_idx] = True
    for chunk in np.array_split(np.asarray(centre_idx), max(1, len(centre_idx) // 256)):
        d = x[None, :, :] - x[chunk][:, None, :]
        f = d @ inv
        d = (f - np.round(f)) @ cell
        near |= (np.linalg.norm(d, axis=-1) < radius).any(0)
    return np.flatnonzero(near)


class RegionTeacher:
    """MACE forces on the region's atoms from a subsystem (region + buffer), masking site energies beyond ``inner``.

    The subsystem is chosen once from the starting positions (fixed atom sets: an interface whose regions do not mix on the
    run's time scale, e.g. a liquid film on a solid). Eager, not graph-captured: v1.
    """

    def __init__(self, calc, atoms, region_idx, buffer: float, inner: float, device: str = "cuda"):
        from frost.calibrate import _batch_for
        from frost.neighbours import VerletList

        self.model = calc.models[0]
        self.r_max = float(self.model.r_max)
        x = atoms.get_positions()
        cell = np.array(atoms.cell.array, dtype=np.float64)
        region_idx = np.asarray(sorted(region_idx), dtype=np.int64)
        self.sub_idx = atoms_within(x, cell, region_idx, buffer)
        pos_in_sub = {int(a): k for k, a in enumerate(self.sub_idx)}
        self.region_in_sub = torch.as_tensor([pos_in_sub[int(a)] for a in region_idx], device=device)
        inner_idx = atoms_within(x, cell, region_idx, inner)
        mask = np.isin(self.sub_idx, inner_idx)
        self.inner_mask = torch.as_tensor(mask, device=device)
        sub = atoms[self.sub_idx]
        self.template = _batch_for(calc, sub, self.r_max, device)[0].to_dict()
        self.dtype = self.template["positions"].dtype
        self.list = VerletList(cell, self.r_max, 1.0, device, wrap_positions=False, builder="alchemi")
        self.sub_t = torch.as_tensor(self.sub_idx, device=device)
        self.n_sub, self.n_region, self.n_inner = len(self.sub_idx), len(region_idx), int(mask.sum())
        self.calls = 0

    def __call__(self, x_full: torch.Tensor) -> torch.Tensor:
        """Forces (n_region, 3) on the region's atoms, float64."""
        xs = x_full[self.sub_t].detach()
        self.list.update(xs)
        data = dict(self.template)
        pos = xs.to(self.dtype).clone().requires_grad_(True)
        data["positions"] = pos
        data["edge_index"] = torch.stack((self.list.i, self.list.j))
        data["shifts"] = self.list.shifts.to(self.dtype)
        data["unit_shifts"] = self.list.unit_shifts.to(self.dtype)
        out = self.model(data, compute_force=False, compute_virials=False, compute_stress=False, training=False)
        e = out["node_energy"][self.inner_mask].sum()
        (g,) = torch.autograd.grad(e, pos)
        self.calls += 1
        return (-g[self.region_in_sub]).double().detach()


class GraphedRegionTeacher(RegionTeacher):
    """``RegionTeacher`` with its masked force call captured in a CUDA graph (2026-11-15).

    The eager version builds a new autograd graph per call, which cost more wall time than the graph-captured whole-cell
    teacher even when it evaluated 1.84x fewer atoms (the 8 A buffer on the 1,170-atom Cu(111)/water cell: 1.78x against
    the global interval's 2.18x). Here the subsystem's positions, edges and inner-atom weights are static buffers; the
    capture holds MACE's forward (its public ``node_energy`` output), the masked sum and one ``autograd.grad`` to the
    positions, as ``GraphedTeacher``'s plain path holds MACE's own force call. Edges live in a fixed-capacity buffer whose
    spare slots are spread over the subsystem's atoms and pushed beyond the cutoff (CLAUDE.md traps 9 and 29), so a list
    rebuild rewrites the buffers in place and recaptures only when the list outgrows the capacity. Same results as the
    eager version to float32 round-off (checked at the start of every ``spatial_runtime.py --graphed-region`` run)."""

    def __init__(self, calc, atoms, region_idx, buffer: float, inner: float, device: str = "cuda", margin: float = 1.1,
                 edge_path: bool = False, tight: bool = False):
        super().__init__(calc, atoms, region_idx, buffer, inner, device)
        # tight (2026-11-17): the subsystem's skin list filtered to r_max before every call (frost.neighbours.tight_edges),
        # as the runtime's tight teacher: the region profile found its skin edges (135,022 against 82,518 within r_max)
        # behind most of the region call's excess over a plain call on the same atoms
        self.tight = tight
        # edge_path (2026-11-15): the masked energy's gradient to the edge vectors, summed onto atoms with index_add
        # (frost.mace_compose.masked_forces_by_edges), not through the sort-based backward of the position gather
        self.edge_path = edge_path
        self.margin = margin
        self.capacity = 0
        self.graph = None
        self.captures = 0
        self.weights = self.inner_mask.to(self.dtype)
        self.version = -1

    def _allocate(self, capacity: int) -> None:
        dev = self.template["positions"].device
        self.capacity = capacity
        self.data = dict(self.template)
        self.data["positions"] = torch.zeros(self.n_sub, 3, dtype=self.dtype, device=dev, requires_grad=True)
        self.data["edge_index"] = torch.zeros(2, capacity, dtype=torch.long, device=dev)
        self.data["shifts"] = torch.zeros(capacity, 3, dtype=self.dtype, device=dev)
        self.data["unit_shifts"] = torch.zeros(capacity, 3, dtype=self.dtype, device=dev)
        self.graph = None

    def _call(self):
        if self.edge_path:
            from frost.mace_compose import masked_forces_by_edges

            return masked_forces_by_edges(self.model, self.data, self.weights)
        out = self.model(self.data, compute_force=False, compute_virials=False, compute_stress=False, training=False)
        e = (out["node_energy"].to(self.dtype) * self.weights).sum()
        (g,) = torch.autograd.grad(e, self.data["positions"])
        return -g

    def _capture(self) -> None:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())  # trap 12: the static inputs were just written
        with torch.cuda.stream(side):
            for _ in range(3):
                self._call()
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_forces = self._call()
        self.captures += 1

    def _set_edges(self, edges_in=None) -> None:
        from frost.padding import spread_padding
        from frost.teacher_fast import _FAR

        i, j, shifts, unit_shifts = edges_in if edges_in is not None else (
            self.list.i, self.list.j, self.list.shifts, self.list.unit_shifts)
        n = int(i.numel())
        if n > self.capacity:
            self._allocate(int(self.margin * n) + 1)
        with torch.no_grad():
            edges = self.data["edge_index"]
            edges[0, :n] = i
            edges[1, :n] = j
            pad = spread_padding(n, self.capacity, self.n_sub, edges.device)
            edges[0, n:] = pad
            edges[1, n:] = pad
            sh = self.data["shifts"]
            sh.zero_()
            sh[n:, 0] = _FAR
            sh[:n] = shifts.to(self.dtype)
            us = self.data["unit_shifts"]
            us.zero_()
            us[:n] = unit_shifts.to(self.dtype)

    def __call__(self, x_full: torch.Tensor) -> torch.Tensor:
        """Forces (n_region, 3) on the region's atoms, float64 (a copy, not the graph's static output)."""
        xs = x_full[self.sub_t].detach()
        self.list.update(xs)
        if self.tight:
            from frost.neighbours import tight_edges

            self._set_edges(tight_edges(self.list, xs, self.r_max ** 2))
        elif self.list.rebuilds != self.version:
            self.version = self.list.rebuilds
            self._set_edges()
        with torch.no_grad():
            self.data["positions"].copy_(xs.to(self.dtype))
        if self.graph is None:
            self._capture()
        self.graph.replay()
        self.calls += 1
        return self.static_forces[self.region_in_sub].detach().double().clone()
