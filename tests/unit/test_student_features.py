"""``MaceStudent.node_features``: the invariant channels a C5 residual head is built on."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from frost.neighbours import edges_for

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")  # e3nn 0.4.4 on torch >= 2.6
pytest.importorskip("mace")

from frost.student_mace import MaceStudent

CUTOFF = 4.0


def setup(layers: int):
    rng = np.random.default_rng(0)
    cell = np.diag([12.0, 12.3, 12.6])
    grid = np.stack(np.meshgrid(*(np.arange(4),) * 3, indexing="ij"), -1).reshape(-1, 3)
    x = grid * np.array([3.0, 3.075, 3.15]) + rng.normal(0.0, 0.2, grid.shape)
    numbers = np.array([14, 8] * 32)
    torch.manual_seed(0)
    model = MaceStudent([8, 14], cutoff=CUTOFF, channels=8, max_ell=1, layers=layers,
                        correlation=2, mean_neighbours=6.0).double()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1 * torch.randn_like(p))
    types = model.types(torch.as_tensor(numbers))
    xw, i, j, sh, _ = edges_for(x, cell, CUTOFF, dtype=torch.float64)
    return model, types, xw, i, j, sh


@pytest.mark.parametrize("layers", [1, 2])
def test_node_features_are_the_invariant_channels_of_every_layer(layers):
    model, types, x, i, j, sh = setup(layers)
    h = model.node_features(x, types, i, j, sh)
    assert h.shape == (x.shape[0], 8 * layers)
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64,
                                       generator=torch.Generator().manual_seed(2)))
    h_rot = model.node_features(x @ q.T, types, i, j, sh @ q.T)
    assert torch.allclose(h, h_rot, atol=1e-10)


def test_a_linear_head_on_node_features_is_a_conservative_force():
    model, types, x, i, j, sh = setup(2)
    w = torch.randn(16, dtype=torch.float64, generator=torch.Generator().manual_seed(3))

    def head_energy(y):
        return (model.node_features(y, types, i, j, sh).sum(0) * w).sum()

    y = x.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(head_energy(y), y)
    direction = torch.randn(x.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    step = 1e-5
    numeric = (head_energy(x + step * direction) - head_energy(x - step * direction)) / (2 * step)
    assert float((grad * direction).sum()) == pytest.approx(float(numeric), rel=1e-5)
