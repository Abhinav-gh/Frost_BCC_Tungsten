"""Interface pinning (frost.pinning): the order parameter, its forces, independence from L_z, and the melting-point fit."""

from __future__ import annotations

import math

import numpy as np
import torch
from ase.build import bulk

from frost.pinning import Pinning, block_se, delta_mu, inplane_kvectors, melting_point


def _fcc(n=(4, 4, 3), a=3.6):
    at = bulk("Cu", "fcc", a=a, cubic=True).repeat(n)
    return at, np.array(at.cell)


def test_perfect_crystal_has_q_one_and_random_positions_near_zero():
    at, cell = _fcc()
    k = inplane_kvectors(cell, (4, 4))
    assert np.allclose(k[0], [4 * math.pi / 3.6, 0, 0]) and np.allclose(k[1], [0, 4 * math.pi / 3.6, 0])
    pin = Pinning(k, kappa=100.0, a=0.5, n_atoms=len(at))
    assert abs(float(pin.q(torch.as_tensor(at.get_positions()))) - 1.0) < 1e-12
    rng = np.random.default_rng(0)
    x = torch.as_tensor(rng.uniform(0, 1, (len(at), 3)) @ cell)
    assert float(pin.q(x)) < 4.0 / math.sqrt(len(at))  # a liquid's Q is O(N^-1/2)


def test_forces_are_minus_the_gradient():
    at, cell = _fcc((3, 3, 2))
    rng = np.random.default_rng(1)
    x0 = at.get_positions() + rng.normal(0, 0.3, (len(at), 3))
    pin = Pinning(inplane_kvectors(cell, (3, 3)), kappa=250.0, a=0.3, n_atoms=len(at))
    _, f, _ = pin(torch.as_tensor(x0))
    h = 1e-5
    for j, d in [(0, 0), (5, 1), (17, 2), (40, 0)]:
        xp, xm = x0.copy(), x0.copy()
        xp[j, d] += h
        xm[j, d] -= h
        num = -(float(pin(torch.as_tensor(xp))[0]) - float(pin(torch.as_tensor(xm))[0])) / (2 * h)
        assert abs(num - float(f[j, d])) < 1e-7 * max(1.0, abs(num)), (j, d, num, float(f[j, d]))


def test_q_ignores_the_z_axis_so_the_bias_does_no_work_on_a_z_barostat():
    at, cell = _fcc()
    rng = np.random.default_rng(2)
    x = at.get_positions() + rng.normal(0, 0.2, (len(at), 3))
    pin = Pinning(inplane_kvectors(cell, (4, 4)), kappa=100.0, a=0.5, n_atoms=len(at))
    y = x.copy()
    y[:, 2] *= 1.037
    assert abs(float(pin.q(torch.as_tensor(x))) - float(pin.q(torch.as_tensor(y)))) < 1e-12
    assert float(pin(torch.as_tensor(x))[1][:, 2].abs().max()) == 0.0
    try:
        inplane_kvectors(cell, (4, 4), hkl=((1, 1, 1),))
        raise AssertionError("an out-of-plane vector must be refused")
    except ValueError:
        pass


def test_delta_mu_sign_and_melting_point_fit():
    # above T_m the solid's chemical potential is higher: the pinned Q sits below a, and dmu > 0
    assert delta_mu(300.0, 0.45, 0.43, 0.88, 0.02, 2880) > 0
    t = np.array([1300.0, 1350.0, 1400.0])
    gaps = 2e-4 * (t - 1362.0)
    tm, se = melting_point(t, gaps, np.full(3, 1e-3))
    assert abs(tm - 1362.0) < 1e-6 and 0 < se < 50


def test_block_se_on_white_noise():
    rng = np.random.default_rng(3)
    v = rng.normal(0, 1, 20000)
    assert abs(block_se(v) / (1 / math.sqrt(len(v))) - 1) < 0.6
