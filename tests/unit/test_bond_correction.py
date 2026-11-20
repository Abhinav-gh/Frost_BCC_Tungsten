"""The bond-stiffness correction: a conservative harmonic term whose online fit recovers a missing stiffness."""

from __future__ import annotations

import numpy as np
import torch

from frost.bond_correction import BondStiffness


def water_like(seed: int = 0):
    rng = np.random.default_rng(seed)
    cell = np.diag([8.0, 8.0, 8.0])
    o = rng.uniform(0.0, 8.0, (6, 3))
    h1 = o + np.array([0.96, 0.0, 0.0]) + rng.normal(0, 0.03, (6, 3))
    h2 = o + np.array([-0.24, 0.93, 0.0]) + rng.normal(0, 0.03, (6, 3))
    x = np.concatenate([o, h1, h2]) % 8.0  # wrapped: some bonds cross the cell boundary
    numbers = np.array([8] * 6 + [1] * 12)
    return numbers, torch.as_tensor(x, dtype=torch.float64), cell


def test_forces_are_the_negative_energy_gradient_across_the_boundary():
    numbers, x, cell = water_like()
    idx, kind = BondStiffness.detect(numbers, x.numpy(), cell, ((1, 8),))
    assert len(idx) == 12
    bond = BondStiffness(idx, kind, x, cell, 1)
    bond.dk.fill_(7.5)
    xp = x + 0.02 * torch.randn_like(x, generator=torch.Generator().manual_seed(1))
    _, f = bond.energy_forces(xp)
    d = torch.randn_like(x, generator=torch.Generator().manual_seed(2))
    h = 1e-6
    numeric = -(bond.energy_forces(xp + h * d)[0] - bond.energy_forces(xp - h * d)[0]) / (2 * h)
    assert abs(float((f * d).sum()) - float(numeric)) < 1e-6


def test_online_fit_recovers_the_missing_stiffness():
    numbers, x, cell = water_like()
    idx, kind = BondStiffness.detect(numbers, x.numpy(), cell, ((1, 8),))
    k_true = 4.0
    teacher = BondStiffness(idx, kind, x, cell, 1)
    teacher.dk.fill_(k_true)  # the stiffness the student lacks
    bond = BondStiffness(idx, kind, x, cell, 1, memory=0.5)
    gen = torch.Generator().manual_seed(3)
    x_old = x.clone()
    r_old = teacher.energy_forces(x_old)[1] - bond.energy_forces(x_old)[1]
    for _ in range(25):
        x_new = x + 0.03 * torch.randn(x.shape, generator=gen, dtype=x.dtype)
        r_new = teacher.energy_forces(x_new)[1] - bond.energy_forces(x_new)[1]
        bond.update(r_new - r_old, x_old, x_new)
        x_old, r_old = x_new, teacher.energy_forces(x_new)[1] - bond.energy_forces(x_new)[1]
    assert abs(float(bond.dk[0]) - k_true) < 0.05 * k_true
