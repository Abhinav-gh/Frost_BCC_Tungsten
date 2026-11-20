"""A force-only (non-conservative) student: DMTS-NC's idea on Frost's own environment features.

DMTS-NC (Gouraud et al., JCTC 22, 5957, 2026) replaces the distilled student's energy with a model that predicts
forces directly: no backward pass, so a cheaper inner step, at the price of forces that are not the gradient of any
energy. Their student is rotation-equivariant and its forces sum to zero over every connected component of the
interaction graph. Their weights and code are not public; this is our implementation of that contract.

The force on atom i is a sum of pair terms along its bonds,

    F_i = sum_j w_ij u_ij,   u_ij = (x_j - x_i + shift) / |...|,

with w_ij a learned scalar of the two atoms' environments and the bond length, built symmetric in (i, j): the
node codes enter as h_i + h_j and h_i * h_j, the species pair as an unordered embedding, the length through the
radial basis. Each bond then contributes +w u to one atom and -w u to the other (u_ji = -u_ij), so Newton's third
law holds pair by pair and the net force is exactly zero; F rotates with the frame because w is invariant and u
rotates. The environment features are ``FrostStudent.node_features`` (body order three, l <= 2 or 3), and the
envelope and mask multiply w, so a force vanishes smoothly at the cutoff and padded or skin edges contribute
nothing.

Non-conservative forces are unsafe alone and safe under multiple time-stepping with a conservative correction
(Bigi, Langer, Ceriotti, ICML 2025). In Frost the teacher is that correction, and the certificate measures what is
left: the drift clause in NVE, the bias clause under a thermostat.

``returns_forces`` tells the callers (``energy_and_forces``, ``GraphedStudent``, ``CompiledGraphedStudent``,
``InSituTrainer``) that ``forward`` returns forces (N, 3), not energies; its energy is reported as zero.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from frost.student import FrostStudent


class ForceStudent(nn.Module):
    returns_forces = True

    def __init__(self, species: Sequence[int], cutoff: float = 4.0, n_rbf: int = 8, channels: int = 16,
                 mix: int = 16, hidden: int = 64, envelope_p: int = 6, lmax: int = 2, body4: bool = False,
                 node_dim: int = 16, edge_hidden: int = 32):
        super().__init__()
        self.env = FrostStudent(species, cutoff=cutoff, n_rbf=n_rbf, channels=channels, mix=mix, hidden=hidden,
                                envelope_p=envelope_p, lmax=lmax, body4=body4)
        n_feat = channels + lmax * mix + 8 + (2 * mix if body4 else 0)
        self.node = nn.Sequential(nn.Linear(n_feat, hidden), nn.SiLU(), nn.Linear(hidden, node_dim))
        n_species = self.env.n_species
        self.pair = nn.Embedding(n_species * n_species, 8)
        self.edge = nn.Sequential(nn.Linear(2 * node_dim + n_rbf + 8, edge_hidden), nn.SiLU(),
                                  nn.Linear(edge_hidden, edge_hidden), nn.SiLU(), nn.Linear(edge_hidden, 1))
        with torch.no_grad():
            self.edge[-1].weight.mul_(0.1)
            self.edge[-1].bias.zero_()
        self.config = {**self.env.config, "kind": "nc", "node_dim": node_dim, "edge_hidden": edge_hidden}

    @property
    def species(self):
        return self.env.species

    def types(self, numbers: torch.Tensor) -> torch.Tensor:
        return self.env.types(numbers)

    def set_neighbour_normalisation(self, mean_neighbours: float) -> None:
        self.env.set_neighbour_normalisation(mean_neighbours)

    def forward(self, x, types, edge_i, edge_j, shifts, batch=None, n_graphs: int = 1, edge_mask=None):
        """Forces (N, 3), eV/A. ``batch`` and ``n_graphs`` are accepted for the energy students' signature."""
        u, _, basis, envelope = self.env.edge_basis(x, edge_i, edge_j, shifts, edge_mask)
        h = self.node(self.env.node_features(types, edge_i, edge_j, u, basis, x.shape[0]))
        hi, hj = h.index_select(0, edge_i), h.index_select(0, edge_j)
        ti, tj = types.index_select(0, edge_i), types.index_select(0, edge_j)
        n = self.env.n_species
        pair = self.pair(torch.minimum(ti, tj) * n + torch.maximum(ti, tj))  # unordered: symmetric in (i, j)
        w = self.edge(torch.cat((hi + hj, hi * hj, basis, pair), -1)).squeeze(-1) * envelope
        fw = w.unsqueeze(-1) * u  # +w u on i from edge (i, j), and -w u on j from (j, i): zero net force
        return fw.new_zeros(x.shape[0], 3).index_add(0, edge_i, fw)
