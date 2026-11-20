"""ZBL repulsion (frost.zbl): the known W-W value at 1 A, the switch's ends, and forces equal to minus the energy gradient."""

from __future__ import annotations

import torch

from frost.zbl import ZBL, switch, zbl_pair


def test_zbl_w_w_at_one_angstrom_and_the_switch():
    v = float(zbl_pair(torch.tensor(1.0, dtype=torch.float64), torch.tensor(74.0, dtype=torch.float64),
                       torch.tensor(74.0, dtype=torch.float64)))
    # by hand: a = 0.4685 / (2 x 74^0.23) = 0.0871 A, x = 11.48, phi = 0.00554, V = 14.4 x 5476 x 0.00554 = ~437 eV
    assert abs(v - 437.0) < 3.0
    r = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5], dtype=torch.float64)
    s = switch(r, 1.0, 2.0)
    assert float(s[0]) == 1.0 and float(s[1]) == 1.0 and abs(float(s[2]) - 0.5) < 1e-12 and float(s[3]) == 0.0


def test_zbl_forces_are_minus_the_energy_gradient():
    torch.manual_seed(0)
    x = torch.tensor([[0.0, 0.0, 0.0], [1.3, 0.1, 0.0], [0.2, 1.4, 0.3]], dtype=torch.float64)
    numbers = torch.tensor([74, 74, 26])
    i = torch.tensor([0, 1, 0, 2, 1, 2])
    j = torch.tensor([1, 0, 2, 0, 2, 1])
    shifts = torch.zeros(6, 3, dtype=torch.float64)
    zbl = ZBL(numbers, 1.0, 2.0)
    e, f = zbl(x, i, j, shifts)
    h = 1e-6
    for a in range(3):
        for c in range(3):
            xp, xm = x.clone(), x.clone()
            xp[a, c] += h
            xm[a, c] -= h
            num = -(float(zbl(xp, i, j, shifts)[0]) - float(zbl(xm, i, j, shifts)[0])) / (2 * h)
            assert abs(num - float(f[a, c])) < 1e-5 * max(1.0, abs(num))
    assert float(f.sum(0).abs().max()) < 1e-9  # pair forces: no net force
