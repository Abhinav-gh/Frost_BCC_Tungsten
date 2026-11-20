"""HeldLongRange (frost.les_split) on a tiny, randomly initialised MACELES on the CPU: the composed forward reproduces
MACELES's own energy and forces, the held mode is exact at the refresh point, its energy error away from it is second
order in the displacement, and its force error is several times smaller than holding the whole long-range force."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

OVERLAY = Path(__file__).resolve().parents[2] / ".tools" / "les-overlay"
if OVERLAY.is_dir() and str(OVERLAY) not in sys.path:
    sys.path.insert(0, str(OVERLAY))  # the les library lives beside the env, not in it (les_profile.py)
pytest.importorskip("les")


@pytest.fixture(scope="module")
def setup():
    from ase import Atoms
    from e3nn import o3
    from mace import data as mdata
    from mace import modules
    from mace.modules.extensions import MACELES
    from mace.tools import torch_geometric, utils

    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    z = [1, 8]
    model = MACELES(
        atomic_inter_scale=1.0, atomic_inter_shift=0.0, r_max=4.0, num_bessel=6, num_polynomial_cutoff=5, max_ell=2,
        interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes["RealAgnosticInteractionBlock"], num_interactions=2,
        num_elements=2, hidden_irreps=o3.Irreps("8x0e + 8x1o"), MLP_irreps=o3.Irreps("8x0e"),
        atomic_energies=np.zeros(2), avg_num_neighbors=8.0, atomic_numbers=z, correlation=2,
        gate=torch.nn.functional.silu, radial_MLP=[16, 16])
    model = model.double().eval()
    with torch.no_grad():  # random readouts are tiny; make the charges large enough to matter
        for p in model.les_readouts.parameters():
            p.mul_(20.0)
    for p in model.parameters():
        p.requires_grad_(False)
    rng = np.random.default_rng(1)
    grid = np.stack(np.meshgrid(*[np.arange(3)] * 3, indexing="ij"), -1).reshape(-1, 3) * 2.6 + 0.5
    pos = grid + rng.normal(0, 0.15, grid.shape)
    numbers = np.array([8 if i % 3 == 0 else 1 for i in range(len(pos))])
    atoms = Atoms(numbers=numbers, positions=pos, cell=np.eye(3) * 7.8, pbc=True)
    table = utils.AtomicNumberTable(z)

    def batch(positions):
        a = atoms.copy()
        a.positions = positions
        cfg = mdata.config_from_atoms(a)
        ad = mdata.AtomicData.from_config(cfg, z_table=table, cutoff=4.0, heads=["Default"])
        return torch_geometric.Batch.from_data_list([ad]).to_dict()

    yield model, atoms, batch
    torch.set_default_dtype(previous)


def test_exact_mode_is_maceles(setup):
    from frost.les_split import HeldLongRange

    model, atoms, batch = setup
    ref = model(batch(atoms.positions), compute_force=True)
    held = HeldLongRange(model)
    out = held(batch(atoms.positions))
    assert abs(float(out["energy"].detach() - ref["energy"].detach())) < 1e-10
    assert torch.allclose(out["forces"], ref["forces"], atol=1e-10)
    assert abs(float(ref["les_energy"].detach())) > 1e-3  # the long-range term is not trivially zero


def test_held_mode_is_exact_at_the_refresh_point(setup):
    from frost.les_split import HeldLongRange

    model, atoms, batch = setup
    ref = model(batch(atoms.positions), compute_force=True)
    held = HeldLongRange(model)
    held.refresh(batch(atoms.positions))
    held.mode = "held"
    out = held(batch(atoms.positions))
    assert abs(float(out["energy"] - ref["energy"])) < 1e-10
    assert torch.allclose(out["forces"], ref["forces"], atol=1e-10)
    both = held.exact_and_refresh(batch(atoms.positions))
    assert torch.allclose(both["forces"], ref["forces"], atol=1e-10)
    assert abs(float(both["energy"] - ref["energy"])) < 1e-10


def test_held_energy_is_second_order_and_its_force_beats_holding_the_whole_force(setup):
    """E~ is first order in the displacement, so its energy error is second order and its force error first order; the
    force error is several times smaller than holding the whole long-range force, whose charge path the held mode
    recomputes (4.8x on this model, a random direction, 0.01-0.08 A)."""
    from frost.les_split import HeldLongRange

    model, atoms, batch = setup
    held = HeldLongRange(model)
    x0 = atoms.positions.copy()
    held.refresh(batch(x0))
    held.mode = "held"
    short = HeldLongRange(model)
    short.mode = "short"
    f_lr0 = model(batch(x0), compute_force=True)["forces"] - short(batch(x0))["forces"]
    u = np.random.default_rng(2).normal(size=x0.shape)
    u /= np.linalg.norm(u)
    e_held, f_held, f_whole = [], [], []
    for delta in (0.04, 0.02, 0.01):
        x = x0 + delta * u
        ref = model(batch(x), compute_force=True)
        out = held(batch(x))
        e_held.append(abs(float(out["energy"].detach() - ref["energy"].detach())))
        f_held.append(float((out["forces"] - ref["forces"]).norm()))
        f_whole.append(float((short(batch(x))["forces"] + f_lr0 - ref["forces"]).norm()))
    ratio = lambda v: [v[i] / v[i + 1] for i in range(len(v) - 1)]  # noqa: E731
    assert all(3.5 < r < 4.5 for r in ratio(e_held)), e_held  # quadratic
    assert all(1.8 < r < 2.2 for r in ratio(f_held)), f_held  # linear
    assert all(1.8 < r < 2.2 for r in ratio(f_whole)), f_whole
    assert all(w > 2.5 * h for w, h in zip(f_whole, f_held)), (f_whole, f_held)
