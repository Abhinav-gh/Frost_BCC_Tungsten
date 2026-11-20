"""frost.confT: the configurational temperature reads the temperature configurations were drawn at."""

import numpy as np
import torch

from frost.confT import KB_EV, conf_temperature, grad_and_trace


def _quartic_samples(n, kT, a, b, rng):
    xs = np.linspace(-2.0, 2.0, 40001)
    w = np.exp(-(a * xs ** 2 + b * xs ** 4) / kT)
    c = np.cumsum(w)
    return np.interp(rng.random(n), c / c[-1], xs)


def _t_conf(T_draw, frames=40, seed=0):
    a, b = 0.5, 0.3  # eV/A^2, eV/A^4: an anharmonic well, so the test is not a harmonic identity
    rng = np.random.default_rng(seed)
    g2s, trs = [], []
    for _ in range(frames):
        x = torch.as_tensor(_quartic_samples(3000, KB_EV * T_draw, a, b, rng), dtype=torch.float64).reshape(1000, 3)
        g2, tr, _ = grad_and_trace(lambda q: (a * q ** 2 + b * q ** 4).sum(), x, probes=2,
                                   generator=torch.Generator().manual_seed(1))
        g2s.append(g2)
        trs.append(tr)
    return conf_temperature(g2s, trs)


def test_canonical_configurations_read_their_temperature():
    t, se = _t_conf(300.0)
    assert np.isfinite(se) and se < 3.0
    assert abs(t - 300.0) < 4 * se + 1.0


def test_configurations_drawn_hotter_read_hotter():
    t300, se300 = _t_conf(300.0, seed=2)
    t330, se330 = _t_conf(330.0, seed=3)
    assert abs(t330 - 330.0) < 4 * se330 + 1.0
    assert t330 - t300 > 20.0


def test_hutchinson_trace_matches_the_exact_laplacian_of_a_coupled_potential():
    # a non-separable potential: U = 1/2 x^T A x with A dense; tr(H) = tr(A), the estimate's mean over probes -> tr(A)
    g = torch.Generator().manual_seed(0)
    m = torch.randn(30, 30, generator=g, dtype=torch.float64)
    A = m @ m.T / 30 + torch.eye(30, dtype=torch.float64)
    x = torch.randn(10, 3, generator=g, dtype=torch.float64)
    _, tr, vals = grad_and_trace(lambda q: 0.5 * q.reshape(-1) @ A @ q.reshape(-1), x, probes=400,
                                 generator=torch.Generator().manual_seed(5))
    assert abs(tr - float(torch.trace(A))) < 0.05 * float(torch.trace(A))
