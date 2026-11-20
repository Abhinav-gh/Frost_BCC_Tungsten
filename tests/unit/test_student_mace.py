"""The MACE student keeps the FrostStudent contract: conservative, invariant, padding-safe."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from frost.neighbours import edges_for
from frost.student import energy_and_forces

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")  # e3nn 0.4.4 on torch >= 2.6
pytest.importorskip("mace")

from frost.student_mace import MaceStudent

CUTOFF = 4.0


def setup(seed: int = 0):
    """A tiny MACE on 64 atoms: cheap enough for the CPU test suite."""
    rng = np.random.default_rng(seed)
    cell = np.diag([12.0, 12.3, 12.6])
    grid = np.stack(np.meshgrid(*(np.arange(4),) * 3, indexing="ij"), -1).reshape(-1, 3)
    x = grid * np.array([3.0, 3.075, 3.15]) + rng.normal(0.0, 0.2, grid.shape)
    numbers = np.array([14, 8] * 32)
    torch.manual_seed(seed)
    model = MaceStudent([8, 14], cutoff=CUTOFF, channels=8, max_ell=1, layers=1,
                        correlation=2, mean_neighbours=6.0).double()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1 * torch.randn_like(p))
    types = model.types(torch.as_tensor(numbers))
    xw, i, j, sh, _ = edges_for(x, cell, CUTOFF, dtype=torch.float64)
    return model, types, xw, i, j, sh


def run(model, types, x, i, j, sh, **kw):
    e, f = energy_and_forces(model, x, types, i, j, sh, **kw)
    return e.detach(), f.detach()


def test_mace_student_forces_are_the_negative_energy_gradient():
    model, types, x, i, j, sh = setup()
    _, f = run(model, types, x, i, j, sh)
    direction = torch.randn(x.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    h = 1e-5
    e_plus, _ = run(model, types, x + h * direction, i, j, sh)
    e_minus, _ = run(model, types, x - h * direction, i, j, sh)
    numeric = -(e_plus - e_minus).item() / (2 * h)
    assert (f * direction).sum().item() == pytest.approx(numeric, rel=1e-5)


def test_mace_student_is_rotation_invariant():
    model, types, x, i, j, sh = setup()
    e, f = run(model, types, x, i, j, sh)
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64,
                                       generator=torch.Generator().manual_seed(2)))
    e_rot, f_rot = run(model, types, x @ q.T, i, j, sh @ q.T)
    assert e_rot.item() == pytest.approx(e.item(), rel=1e-8)
    assert torch.allclose(f_rot, f @ q.T, rtol=1e-6, atol=1e-8)


def test_mace_student_ignores_masked_padding():
    model, types, x, i, j, sh = setup()
    pad = 20
    zeros = torch.zeros(pad, dtype=torch.long)
    mask = torch.cat((torch.ones(i.numel()), torch.zeros(pad))).double()
    e, f = run(model, types, x, i, j, sh)
    e_pad, f_pad = run(model, types, x, torch.cat((i, zeros)), torch.cat((j, zeros)),
                       torch.cat((sh, torch.zeros(pad, 3, dtype=torch.float64))), edge_mask=mask)
    assert e_pad.item() == pytest.approx(e.item(), rel=1e-10)
    assert torch.allclose(f_pad, f, atol=1e-9)


def test_mace_student_edge_force_path_matches_autograd():
    """``energy_and_edge_forces`` (mace's public edge gradients, summed with index_add_) gives the same
    energy and forces as differentiating to the positions, with and without padded edges."""
    model, types, x, i, j, sh = setup()
    e_ref, f_ref = run(model, types, x, i, j, sh)
    xg = x.detach().clone().requires_grad_(True)
    e, f = model.energy_and_edge_forces(xg, types, i, j, sh)
    assert e.detach().sum().item() == pytest.approx(e_ref.sum().item(), rel=1e-12)
    assert torch.allclose(f.detach(), f_ref, atol=1e-10)
    # padded slots: atom 0 to itself, far beyond the cutoff, masked out (GraphedStudent's layout)
    pad = 37
    ip = torch.cat([i, torch.zeros(pad, dtype=i.dtype)])
    jp = torch.cat([j, torch.zeros(pad, dtype=j.dtype)])
    shp = torch.cat([sh, torch.zeros(pad, 3, dtype=sh.dtype)])
    mask = torch.cat([torch.ones(i.numel(), dtype=x.dtype), torch.zeros(pad, dtype=x.dtype)])
    e_p, f_p = model.energy_and_edge_forces(x.detach().clone().requires_grad_(True), types, ip, jp, shp,
                                            edge_mask=mask)
    assert torch.allclose(f_p.detach(), f_ref, atol=1e-10)
