"""The teacher's force and energy calls replayed from CUDA graphs.

A MACE force call launches hundreds of small kernels, and on a 1152-atom cell
their launch overhead is a measurable share of the call. Capturing the call once
and replaying it removes that overhead. The captured teacher serves the plain
teacher baseline (B0) and the multirate loop alike, so both sides of every
speedup run on the same kernels.

The energy-only forward is captured in a second graph. It is the whole teacher
cost of a nested-HMC proposal (`frost.nested`), where one energy call accepts or
rejects a student-drafted trajectory, and it is *more* launch-bound than the
force call: it runs the same hundreds of kernels with no backward pass to
amortise them over. Both graphs replay from the same static input buffers, so a
list rebuild that grows the edge capacity invalidates both.

The edge list lives in static buffers of a fixed capacity. Unused slots join
an atom to itself with a shift far beyond r_max, spread over the atoms
(``frost.padding``: all on atom 0, the backward serialised on them and cost up to
3x at 4,000 atoms). MACE multiplies every radial feature by a polynomial envelope
that is exactly zero beyond r_max (mace.modules.radial.PolynomialCutoff), so these
slots add nothing to the energy or the forces, but they cost compute in proportion
to their number. A longer edge list reallocates the buffers at ``margin`` (1.05x:
a recapture is a few calls, a spare slot costs on every call) and recaptures.
"""

from __future__ import annotations

import torch

from frost.padding import spread_padding

_FAR = 1.0e3  # A; any length beyond r_max works


class GraphedTeacher:
    """``frost.teacher.MaceTeacher`` with its force call captured in a CUDA graph.

    Same interface: ``set_edges(i, j, shifts, unit_shifts)``, then ``__call__(x)``
    returns (energy, forces). The returned forces are the graph's static output;
    copy them before the next call if they must be kept.
    """

    def __init__(self, calc, atoms, device: str = "cuda", margin: float = 1.05, edge_forces: bool = False,
                 virial: bool = False):
        from frost.calibrate import _batch_for

        self.model = calc.models[0]
        self.r_max = float(self.model.r_max)
        self.template = _batch_for(calc, atoms, self.r_max, device)[0].to_dict()
        self.dtype = self.template["positions"].dtype
        self.n_atoms = int(self.template["positions"].shape[0])
        self.margin = margin
        self.capacity = 0
        self.graph = None
        self.energy_graph = None
        self.captures = 0
        self.energy_captures = 0
        self.edge_forces = edge_forces
        self.virial = virial and edge_forces  # only on the edge-force path, and only when a barostat asks

    def _allocate(self, capacity: int) -> None:
        dev = self.template["positions"].device
        self.capacity = capacity
        self.data = dict(self.template)
        self.data["positions"] = torch.zeros(self.n_atoms, 3, dtype=self.dtype, device=dev)
        self.data["edge_index"] = torch.zeros(2, capacity, dtype=torch.long, device=dev)
        self.data["shifts"] = torch.zeros(capacity, 3, dtype=self.dtype, device=dev)
        self.data["unit_shifts"] = torch.zeros(capacity, 3, dtype=self.dtype, device=dev)
        self.graph = None
        self.energy_graph = None  # the new buffers are not the ones it captured

    def _call(self):
        if not self.edge_forces:
            out = self.model(self.data, compute_force=True, training=False)
            return out["energy"], out["forces"]
        # The same forces through mace's public edge-force output: dE/dr per edge, then summed onto atoms.
        # Differentiating to the positions instead runs PyTorch's sort-based backward of the edge gather
        # ``positions[receiver] - positions[sender]`` (indexing_backward_kernel), 29-32% of the whole call on
        # Cu, Li3PO4 and silica (teacher_profile.py); index_add_ does the same sum with atomics.
        out = self.model(self.data, compute_force=False, compute_virials=False, compute_stress=False,
                         compute_edge_forces=True, training=False)
        g = out["edge_forces"]  # +dE/dr_e, r_e = x[receiver] - x[sender] + shift
        edges = self.data["edge_index"]
        forces = torch.zeros(self.n_atoms, 3, dtype=g.dtype, device=g.device)
        forces.index_add_(0, edges[0], g)
        forces.index_add_(0, edges[1], -g)
        if self.virial:  # W = -sum_e r_e . dE/dr_e for a barostat (padded edges sit beyond the cutoff: g = 0 there)
            pos = self.data["positions"]
            r = pos[edges[1]] - pos[edges[0]] + self.data["shifts"]
            self.static_virial_buf = -(r.to(g.dtype) * g).sum()
            # the tensor W_ab = -sum_e r_a dE/dr_b, for a barostat on one axis (NP_zAT: the normal stress drives L_z)
            self.static_virial_tensor_buf = -(r.detach().to(g.dtype).unsqueeze(2) * g.detach().unsqueeze(1)).sum(0)
        return out["energy"], forces

    def _energy_call(self):
        with torch.no_grad():
            return self.model(self.data, compute_force=False, training=False)["energy"]

    def _warm(self, call) -> None:
        """Three warm-up calls on a side stream that waits for the default one (trap 12)."""
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())  # the static inputs were just written there
        with torch.cuda.stream(side):
            for _ in range(3):
                call()
        torch.cuda.current_stream().wait_stream(side)

    def _capture(self) -> None:
        self._warm(self._call)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            # not ``self.energy``: that name is the energy-only method below
            self.static_energy, self.static_forces = self._call()
            self.static_virial = getattr(self, "static_virial_buf", None)  # edge-force path only
            self.static_virial_tensor = getattr(self, "static_virial_tensor_buf", None)
        self.captures += 1

    def _capture_energy(self) -> None:
        self._warm(self._energy_call)
        self.energy_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.energy_graph):
            self.static_potential = self._energy_call()
        self.energy_captures += 1

    def set_edges(self, i: torch.Tensor, j: torch.Tensor, shifts: torch.Tensor,
                  unit_shifts: torch.Tensor) -> None:
        n = int(i.numel())
        if n > self.capacity:
            self._allocate(int(self.margin * n) + 1)
        with torch.no_grad():
            edges = self.data["edge_index"]
            edges[0, :n] = i
            edges[1, :n] = j
            pad = spread_padding(n, self.capacity, self.n_atoms, edges.device)
            edges[0, n:] = pad
            edges[1, n:] = pad
            sh = self.data["shifts"]
            sh.zero_()
            sh[n:, 0] = _FAR
            sh[:n] = shifts.to(self.dtype)
            us = self.data["unit_shifts"]
            us.zero_()
            us[:n] = unit_shifts.to(self.dtype)
        if self.graph is None:
            with torch.no_grad():
                self.data["positions"].copy_(self.template["positions"])
            self._capture()

    def __call__(self, x: torch.Tensor):
        with torch.no_grad():
            self.data["positions"].copy_(x.to(self.dtype))
        self.graph.replay()
        return self.static_energy.detach().sum(), self.static_forces.detach()

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """Energy-only forward on the current edges, replayed from its own graph.

        Captured on the first call, after the positions are written, so the
        warm-up runs on the geometry the caller actually asked for. Capture does
        not execute the work, so the replay below is what produces the value.
        """
        with torch.no_grad():
            self.data["positions"].copy_(x.to(self.dtype))
        if self.energy_graph is None:
            self._capture_energy()
        self.energy_graph.replay()
        return self.static_potential.detach().sum()

    def energy_eager(self, x: torch.Tensor) -> torch.Tensor:
        """Energy-only forward, eager, on the current edges. The graph's reference."""
        with torch.no_grad():
            self.data["positions"].copy_(x.to(self.dtype))
        return self._energy_call().detach().sum()
