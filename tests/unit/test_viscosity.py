"""Green-Kubo pieces of experiments/e13_runtime/viscosity.py on a signal whose integral is known (2026-11-17)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "viscosity", Path(__file__).resolve().parents[2] / "experiments" / "e13_runtime" / "viscosity.py")
visc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(visc)


def test_green_kubo_integrates_an_exponential_correlation():
    """Five independent Ornstein-Uhlenbeck series, unit variance, correlation time 20: the integral is 20."""
    rng = np.random.default_rng(0)
    n, tau, dt = 200_000, 20.0, 1.0
    a = np.exp(-dt / tau)
    x = np.zeros((n, 5))
    noise = rng.normal(size=(n, 5)) * np.sqrt(1 - a * a)
    for i in range(1, n):
        x[i] = a * x[i - 1] + noise[i]
    temp = visc.EV_FS_A3_TO_MPAS / visc.KB  # V / (k_B T) * conversion = 1
    run, c = visc.green_kubo(x, dt, 1.0, temp, 200.0)
    assert abs(c[0] - 1.0) < 0.02
    assert abs(run[-1] - tau) / tau < 0.08


def test_shear_components_are_the_traceless_parts():
    p = np.array([[[1.0, 2.0, 3.0], [2.0, 5.0, 4.0], [3.0, 4.0, 9.0]]])
    assert np.allclose(visc.shear_components(p), [[2.0, 3.0, 4.0, -2.0, -2.0]])
