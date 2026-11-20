"""MACE's forward composed from its own public submodules, up to the node energies (2026-11-15).

Two Frost paths need what MACE's forward computes inside but does not return: the long-range third level
(``frost.les_split``: MACELES's latent charges with the Ewald sum switchable) and the spatial region teacher
(``frost.spatial``: the edge-vector gradient of a *masked* sum of node energies, where MACE's ``compute_edge_forces`` gives
it only for the total energy). Both call the model's submodules in the order ``ScaleShiftMACE.forward`` /
``MACELES.forward`` do, through mace's public helpers (``prepare_graph``, ``scatter_sum``); no library object is touched.
``tests/unit/test_mace_compose.py`` checks the node energies against the model's own and the edge-path forces against
autograd to the positions.
"""

from __future__ import annotations

import torch
from mace.modules.utils import prepare_graph
from mace.tools.scatter import scatter_sum


def node_energies(model, data: dict, charges: bool = False):
    """(ctx, e0, inter_e, node_e0, node_inter, q): mace's GraphContext (positions, edge vectors, ...), the reference
    energies per graph, the interaction energy per graph, the per-node reference and interaction energies, and, with
    ``charges`` (a MACELES), its latent charges. Plain (non-LAMMPS) graphs, no joint embeddings."""
    m = model
    ctx = prepare_graph(data)
    arange, n_graphs, heads = ctx.num_atoms_arange, ctx.num_graphs, ctx.node_heads
    ikw = ctx.interaction_kwargs
    node_e0 = m.atomic_energies_fn(data["node_attrs"])[arange, heads]
    e0 = scatter_sum(src=node_e0, index=data["batch"], dim=0, dim_size=n_graphs).to(ctx.vectors.dtype)
    node_feats = m.node_embedding(data["node_attrs"])
    edge_attrs = m.spherical_harmonics(ctx.vectors)
    edge_feats, cutoff = m.radial_embedding(ctx.lengths, data["node_attrs"], data["edge_index"], m.atomic_numbers)
    if hasattr(m, "pair_repulsion"):
        pair = m.pair_repulsion_fn(ctx.lengths, data["node_attrs"], data["edge_index"], m.atomic_numbers)
    else:
        pair = torch.zeros_like(node_e0)
    if hasattr(m, "joint_embedding"):
        raise NotImplementedError("MACE with joint embeddings is not composed here")
    energies, feats = [pair], []
    for i, (interaction, product) in enumerate(zip(m.interactions, m.products)):
        node_feats, sc = interaction(node_attrs=data["node_attrs"], node_feats=node_feats, edge_attrs=edge_attrs,
                                     edge_feats=edge_feats, edge_index=data["edge_index"], cutoff=cutoff,
                                     first_layer=(i == 0), lammps_class=ikw.lammps_class, lammps_natoms=ikw.lammps_natoms)
        node_feats = product(node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"])
        feats.append(node_feats)
    charge_list = []
    for i, readout in enumerate(m.readouts):
        idx = -1 if len(m.readouts) == 1 else i
        energies.append(readout(feats[idx], heads)[arange, heads])
        if charges:
            charge_list.append(m.les_readouts[i](feats[idx], heads)[arange, heads])
    node_inter = m.scale_shift(torch.sum(torch.stack(energies, dim=0), dim=0), heads)
    inter_e = scatter_sum(node_inter, data["batch"], dim=-1, dim_size=n_graphs)
    q = torch.sum(torch.stack(charge_list, dim=1), dim=1) if charges else None
    return ctx, e0, inter_e, node_e0, node_inter, q


def masked_forces_by_edges(model, data: dict, weights: torch.Tensor) -> torch.Tensor:
    """Forces of E_w = sum_i w_i E_i (node interaction energies) on every node: the gradient taken to the edge vectors
    r_e = x[receiver] - x[sender] + shift and summed onto atoms with index_add (as ``GraphedTeacher``'s edge path does for
    the total energy), not through PyTorch's sort-based backward of the position gather (29-32% of a whole call)."""
    ctx, _, _, _, node_inter, _ = node_energies(model, data)
    e = (node_inter.to(weights.dtype) * weights).sum()
    (g,) = torch.autograd.grad(e, ctx.vectors)  # +dE_w / dr_e
    edges = data["edge_index"]
    forces = torch.zeros(ctx.positions.shape[0], 3, dtype=g.dtype, device=g.device)
    forces.index_add_(0, edges[0], g)
    forces.index_add_(0, edges[1], -g)
    return forces
