"""frost.mace_compose on a tiny, randomly initialised ScaleShiftMACE on the CPU (float64): the composed forward's node
energies are the model's own, and the masked forces summed from edge-vector gradients equal autograd to the positions."""

from __future__ import annotations

import numpy as np
import torch


def _tiny():
    from ase import Atoms
    from e3nn import o3
    from mace import data as mdata
    from mace import modules
    from mace.tools import torch_geometric, utils

    torch.manual_seed(0)
    z = [13, 29]
    model = modules.ScaleShiftMACE(
        atomic_inter_scale=1.3, atomic_inter_shift=0.1, r_max=4.0, num_bessel=6, num_polynomial_cutoff=5, max_ell=2,
        interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes["RealAgnosticInteractionBlock"], num_interactions=2,
        num_elements=2, hidden_irreps=o3.Irreps("8x0e + 8x1o"), MLP_irreps=o3.Irreps("8x0e"),
        atomic_energies=np.array([-1.0, -2.0]), avg_num_neighbors=10.0, atomic_numbers=z, correlation=2,
        gate=torch.nn.functional.silu, radial_MLP=[16, 16]).double().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    rng = np.random.default_rng(3)
    grid = np.stack(np.meshgrid(*[np.arange(3)] * 3, indexing="ij"), -1).reshape(-1, 3) * 2.5 + 0.3
    pos = grid + rng.normal(0, 0.1, grid.shape)
    numbers = np.where(rng.random(len(pos)) < 0.5, 13, 29)
    atoms = Atoms(numbers=numbers, positions=pos, cell=np.eye(3) * 7.5, pbc=True)
    cfg = mdata.config_from_atoms(atoms)
    ad = mdata.AtomicData.from_config(cfg, z_table=utils.AtomicNumberTable(z), cutoff=4.0, heads=["Default"])
    return model, torch_geometric.Batch.from_data_list([ad]).to_dict()


def test_node_energies_are_the_models_own():
    from frost.mace_compose import node_energies

    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        model, data = _tiny()
        ref = model(dict(data), compute_force=False)
        ctx, e0, inter_e, node_e0, node_inter, q = node_energies(model, dict(data))
        assert q is None
        assert torch.allclose(node_e0 + node_inter, ref["node_energy"], atol=1e-12)
        assert torch.allclose((e0 + inter_e).detach(), ref["energy"].detach(), atol=1e-10)
    finally:
        torch.set_default_dtype(previous)


def test_masked_edge_forces_equal_autograd_to_positions():
    from frost.mace_compose import masked_forces_by_edges, node_energies

    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        model, data = _tiny()
        n = data["positions"].shape[0]
        w = torch.zeros(n, dtype=torch.float64)
        w[: n // 2] = 1.0  # a region: the first half of the atoms' site energies
        f_edges = masked_forces_by_edges(model, dict(data), w)
        d = dict(data)
        d["positions"] = data["positions"].detach().clone().requires_grad_(True)
        _, _, _, _, node_inter, _ = node_energies(model, d)
        (g,) = torch.autograd.grad((node_inter * w).sum(), d["positions"])
        assert torch.allclose(f_edges, -g, atol=1e-10)
        assert float(f_edges[n // 2:].abs().max()) > 1e-6  # atoms outside the mask still feel the masked energy
    finally:
        torch.set_default_dtype(previous)
