"""The in-situ student: a small, strictly local, conservative potential.

The reframed design (docs/design/idea_extension_frost.tex, section 5.2) runs a
cheap student every step and refreshes only the teacher's residual
R = F_T - F_S between teacher calls. Proposition 2 of that note says the refresh
interval is set by how well the student tracks the teacher's curvature along the
motion, and the speedup by the student's cost. This model is built for both.

* Strictly local. One message-free layer, so the energy is a sum of site
  energies that each depend only on neighbours within ``cutoff``. Site
  residuals are then closed, output-level units (note section 5.4).
* Body order three with a learned readout. Bessel radial channels, gated by
  species pair, build angular moments of order l = 0..lmax (lmax 2 or 3) around
  each atom. Channel-mixed dot products of the moments are rotation invariant
  and carry bond angles; l = 3 resolves tetrahedral order, which silica needs.
  A small MLP maps them to a site energy.
* Conservative. Forces are -dE/dx by autograd, which keeps RESPA symplectic and
  gives the nested-verification contract a student energy.
* Few large kernels, fixed shapes. Edges beyond the cutoff (a Verlet skin, or
  padding) are masked rather than removed, so a force call can be captured in a
  CUDA graph at a fixed edge capacity (``GraphedStudent``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from frost.padding import spread_padding

# u (x) u - I/3 is stored as (xx, yy, zz, xy, xz, yz); with this metric a dot
# product of two such rows is the full contraction T : T'.
_L2_METRIC = (1.0, 1.0, 1.0, 2.0, 2.0, 2.0)
# real spherical harmonics of order 3 without the common 1/sqrt(pi): orthonormal
# up to that factor, so a plain dot product of two rows is rotation invariant
_C3 = (
    0.25 * math.sqrt(35 / 2),
    0.5 * math.sqrt(105),
    0.25 * math.sqrt(21 / 2),
    0.25 * math.sqrt(7),
    0.25 * math.sqrt(21 / 2),
    0.25 * math.sqrt(105),
    0.25 * math.sqrt(35 / 2),
)
_WIDTH = {2: 10, 3: 17}  # 1 + 3 + 6 (+ 7) per-edge angular components


class FrostStudent(nn.Module):
    def __init__(
        self,
        species: Sequence[int],
        cutoff: float = 4.0,
        n_rbf: int = 8,
        channels: int = 16,
        mix: int = 16,
        hidden: int = 64,
        envelope_p: int = 6,
        lmax: int = 2,
        body4: bool = False,
    ):
        super().__init__()
        if lmax not in _WIDTH:
            raise ValueError("lmax must be 2 or 3")
        z = sorted({int(s) for s in species})
        table = torch.full((119,), -1, dtype=torch.long)
        for k, s in enumerate(z):
            table[s] = k
        self.species = z
        self.n_species = len(z)
        self.channels = channels
        self.lmax = lmax
        self.p = envelope_p
        self.config = {
            "kind": "frost", "species": z, "cutoff": float(cutoff), "n_rbf": n_rbf,
            "channels": channels, "mix": mix, "hidden": hidden, "envelope_p": envelope_p,
            "lmax": lmax, "body4": bool(body4),
        }
        self.body4 = bool(body4)
        self.register_buffer("species_table", table)
        self.register_buffer("cutoff", torch.tensor(float(cutoff)))
        self.register_buffer("freq", torch.arange(1, n_rbf + 1).float() * math.pi)
        self.register_buffer("metric", torch.tensor(_L2_METRIC))
        self.register_buffer("c3", torch.tensor(_C3))
        self.register_buffer("inv_neighbours", torch.tensor(1.0))

        self.radial = nn.Linear(n_rbf, (lmax + 1) * channels, bias=False)
        self.pair_gate = nn.Embedding(self.n_species**2, (lmax + 1) * channels)
        nn.init.ones_(self.pair_gate.weight)
        self.mix = nn.ModuleList(nn.Linear(channels, mix, bias=False) for _ in range(2 * lmax))
        # body order four (NEP4's q_112 and q_222): a^T M b and tr(A B C) of channel-mixed l = 1 and l = 2 moments
        self.mix4 = nn.ModuleList(nn.Linear(channels, mix, bias=False) for _ in range(6)) if self.body4 else None
        self.embed = nn.Embedding(self.n_species, 8)
        self.readout = nn.Sequential(
            nn.Linear(channels + lmax * mix + 8 + (2 * mix if self.body4 else 0), hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.shift = nn.Embedding(self.n_species, 1)
        with torch.no_grad():
            self.readout[-1].weight.mul_(0.1)
            self.shift.weight.zero_()

    def types(self, numbers: torch.Tensor) -> torch.Tensor:
        """Atomic numbers to the student's species indices."""
        t = self.species_table[numbers.long()]
        if bool((t < 0).any()):
            raise ValueError("an atomic number is outside the student's species list")
        return t

    @torch.no_grad()
    def set_neighbour_normalisation(self, mean_neighbours: float) -> None:
        self.inv_neighbours.fill_(1.0 / max(float(mean_neighbours), 1.0))

    def edge_basis(self, x, edge_i, edge_j, shifts, edge_mask=None):
        """(unit vectors u, lengths d, radial basis times envelope and mask, envelope times mask) per edge."""
        # index_select, not x[edge_j]: the same values, but its backward is an atomic index_add where advanced
        # indexing's is PyTorch's sort-based indexing_backward kernel (29-32% of the MACE teacher's force call)
        r = x.index_select(0, edge_j) - x.index_select(0, edge_i) + shifts
        d = torch.sqrt((r * r).sum(-1) + 1e-12)
        u = r / d.unsqueeze(-1)
        s = d / self.cutoff
        weight = (s < 1.0).to(x.dtype)
        if edge_mask is not None:
            weight = weight * edge_mask
        p = self.p
        s_p = s**p
        envelope = (
            1.0
            - 0.5 * (p + 1) * (p + 2) * s_p
            + p * (p + 2) * s_p * s
            - 0.5 * p * (p + 1) * s_p * s * s
        )
        basis = torch.sin(self.freq * s.unsqueeze(-1)) / d.unsqueeze(-1)
        basis = basis * (envelope * weight).unsqueeze(-1)
        return u, d, basis, envelope * weight

    def node_features(self, types, edge_i, edge_j, u, basis, n_atoms: int):
        """Per-atom rotation-invariant features (the readout's input) from the edge basis."""
        radial = self.radial(basis) * self.pair_gate(types[edge_i] * self.n_species + types[edge_j])
        c = self.channels
        ux, uy, uz = u.unbind(-1)
        x2, y2, z2 = ux * ux, uy * uy, uz * uz
        third = 1.0 / 3.0
        quad = torch.stack((x2 - third, y2 - third, z2 - third, ux * uy, ux * uz, uy * uz), -1)
        blocks = [
            radial[:, :c].unsqueeze(-1),
            radial[:, c : 2 * c].unsqueeze(-1) * u.unsqueeze(1),
            radial[:, 2 * c : 3 * c].unsqueeze(-1) * quad.unsqueeze(1),
        ]
        if self.lmax == 3:
            cubic = torch.stack(
                (
                    uy * (3 * x2 - y2),
                    ux * uy * uz,
                    uy * (5 * z2 - 1),
                    uz * (5 * z2 - 3),
                    ux * (5 * z2 - 1),
                    uz * (x2 - y2),
                    ux * (x2 - 3 * y2),
                ),
                -1,
            ) * self.c3
            blocks.append(radial[:, 3 * c :].unsqueeze(-1) * cubic.unsqueeze(1))
        per_edge = torch.cat(blocks, dim=-1)
        moments = per_edge.new_zeros(n_atoms, c, _WIDTH[self.lmax]).index_add(0, edge_i, per_edge)
        moments = moments * self.inv_neighbours

        w = [lin.weight for lin in self.mix]
        m1, m2 = moments[..., 1:4], moments[..., 4:10]
        invariants = [
            moments[..., 0],
            (torch.einsum("nca,dc->nda", m1, w[0]) * torch.einsum("nca,dc->nda", m1, w[1])).sum(-1),
            (torch.einsum("nca,dc->nda", m2, w[2]) * torch.einsum("nca,dc->nda", m2, w[3])
             * self.metric).sum(-1),
        ]
        if self.lmax == 3:
            m3 = moments[..., 10:17]
            invariants.append(
                (torch.einsum("nca,dc->nda", m3, w[4]) * torch.einsum("nca,dc->nda", m3, w[5])).sum(-1)
            )
        if self.body4:
            w4 = [lin.weight for lin in self.mix4]

            def sym(mixed):  # (n, d, 6) traceless-symmetric components -> (n, d, 3, 3)
                xx, yy, zz, xy, xz, yz = mixed.unbind(-1)
                return torch.stack((torch.stack((xx, xy, xz), -1), torch.stack((xy, yy, yz), -1),
                                    torch.stack((xz, yz, zz), -1)), -2)

            a = torch.einsum("nca,dc->nda", m1, w4[0])
            b = torch.einsum("nca,dc->nda", m1, w4[1])
            mq = sym(torch.einsum("nca,dc->nda", m2, w4[2]))
            ma = sym(torch.einsum("nca,dc->nda", m2, w4[3]))
            mb = sym(torch.einsum("nca,dc->nda", m2, w4[4]))
            mc = sym(torch.einsum("nca,dc->nda", m2, w4[5]))
            invariants.append(torch.einsum("nda,ndab,ndb->nd", a, mq, b))           # q_112
            invariants.append(torch.einsum("ndab,ndbc,ndca->nd", ma, mb, mc))        # q_222
        return torch.cat((*invariants, self.embed(types)), dim=-1)

    def site_energies(self, x, types, edge_i, edge_j, shifts, edge_mask=None):
        u, _, basis, _ = self.edge_basis(x, edge_i, edge_j, shifts, edge_mask)
        features = self.node_features(types, edge_i, edge_j, u, basis, x.shape[0])
        return self.readout(features).squeeze(-1) + self.shift(types).squeeze(-1)

    def forward(self, x, types, edge_i, edge_j, shifts, batch=None, n_graphs: int = 1,
                edge_mask=None):
        """Total energy per graph, shape (n_graphs,)."""
        e = self.site_energies(x, types, edge_i, edge_j, shifts, edge_mask)
        if batch is None:
            return e.sum().unsqueeze(0)
        return e.new_zeros(n_graphs).index_add(0, batch, e)


def energy_and_forces(model, x, types, edge_i, edge_j, shifts, batch=None, n_graphs: int = 1,
                      edge_mask=None, create_graph: bool = False):
    """Energies (n_graphs,) and conservative forces -dE/dx; a force-only student (``returns_forces``) gives its forces
    directly and zero energies."""
    if getattr(model, "returns_forces", False):
        f = model(x, types, edge_i, edge_j, shifts, batch, n_graphs, edge_mask)
        return f.new_zeros(n_graphs), f
    x = x.detach().requires_grad_(True)
    energy = model(x, types, edge_i, edge_j, shifts, batch, n_graphs, edge_mask)
    (grad,) = torch.autograd.grad(energy.sum(), x, create_graph=create_graph)
    return energy, -grad


class GraphedStudent:
    """The student's force call captured in a CUDA graph at a fixed edge capacity.

    Inputs are copied into static buffers and the graph is replayed, which
    removes the per-kernel launch overhead that dominates small models on a GPU.
    Parameters are read through their storage, so in-place optimiser updates are
    picked up without recapture. An edge list longer than the capacity triggers
    a recapture at 1.25x. The returned tensors are the static outputs: clone
    them before the next call if they must be kept.
    """

    def __init__(self, model, types: torch.Tensor, n_atoms: int, capacity: int,
                 dtype=torch.float32):
        self.model = model
        self.types = types
        self.n_atoms = n_atoms
        self.dtype = dtype
        self.graph = None
        self._allocate(capacity)

    def _allocate(self, capacity: int) -> None:
        dev = self.types.device
        self.capacity = capacity
        self.x = torch.zeros(self.n_atoms, 3, device=dev, dtype=self.dtype, requires_grad=True)
        self.i = torch.zeros(capacity, dtype=torch.long, device=dev)
        self.j = torch.zeros(capacity, dtype=torch.long, device=dev)
        self.shifts = torch.zeros(capacity, 3, device=dev, dtype=self.dtype)
        self.mask = torch.zeros(capacity, device=dev, dtype=self.dtype)
        self.graph = None

    def _capture(self) -> None:
        def call():
            if getattr(self.model, "returns_forces", False):  # a force-only student (frost.student_nc): no backward
                with torch.no_grad():
                    f = self.model(self.x, self.types, self.i, self.j, self.shifts, edge_mask=self.mask)
                return f.new_zeros(1), f
            if hasattr(self.model, "energy_and_edge_forces"):  # MaceStudent: forces from edge gradients
                return self.model.energy_and_edge_forces(self.x, self.types, self.i, self.j, self.shifts,
                                                         edge_mask=self.mask)
            e = self.model(self.x, self.types, self.i, self.j, self.shifts, edge_mask=self.mask)
            (g,) = torch.autograd.grad(e.sum(), self.x)
            return e, -g

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())  # the static inputs were just written there
        with torch.cuda.stream(side):
            for _ in range(3):
                call()
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.energy, self.forces = call()

    def set_edges(self, i: torch.Tensor, j: torch.Tensor, shifts: torch.Tensor) -> None:
        n = int(i.numel())
        if n > self.capacity:
            self._allocate(int(1.25 * n) + 1)
        with torch.no_grad():
            self.mask.zero_()
            self.shifts.zero_()
            self.i[:n] = i
            self.j[:n] = j
            pad = spread_padding(n, self.capacity, self.n_atoms, self.i.device)  # not all on atom 0: frost.padding
            self.i[n:] = pad
            self.j[n:] = pad
            self.shifts[:n] = shifts.to(self.dtype)
            self.mask[:n] = 1.0
        if self.graph is None:
            self._capture()

    def __call__(self, x: torch.Tensor):
        with torch.no_grad():
            self.x.copy_(x.to(self.dtype))
        self.graph.replay()
        return self.energy, self.forces
