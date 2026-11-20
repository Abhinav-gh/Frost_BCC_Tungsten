"""Staleness statistics: drift, hold, extrapolation and the acceptance proxy."""

from __future__ import annotations

import math

import numpy as np
import pytest

from frost.staleness import drift, energy_stats, hold, lagged, linear, per_mass, rms


def test_per_mass_turns_a_force_residual_into_an_acceleration_one():
    """The same force error is a much larger acceleration error on a light atom."""
    s = np.ones((2, 5, 3, 3))  # windows, frames, atoms, components
    a = per_mass(s, np.array([1.0, 16.0, 28.0]))
    assert a.shape == s.shape
    assert a[..., 0, :] == pytest.approx(1.0)
    assert a[..., 1, :] == pytest.approx(1.0 / 16.0)
    assert a[..., 2, :] == pytest.approx(1.0 / 28.0)
    # a uniform residual converts by the RMS of 1/m: the factor that reads a
    # force-unit staleness from an older run in acceleration units (plan 1.11)
    masses = np.array([1.0, 16.0, 28.0])
    factor = float(np.sqrt(np.mean(1.0 / masses**2)))
    assert rms(per_mass(s, masses)) == pytest.approx(rms(s) * factor)


def ramp(frames: int = 33, atoms: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """A quantity moving at constant velocity, s_t = s_0 + t v, in one window."""
    rng = np.random.default_rng(0)
    v = rng.normal(size=(atoms, 3))
    s0 = rng.normal(size=(atoms, 3))
    t = np.arange(frames)[None, :, None, None]
    return s0 + t * v, v


def test_drift_is_linear_in_k_for_constant_velocity():
    s, v = ramp()
    speed = float(np.sqrt((v**2).sum(-1).mean()))
    for k in (1, 2, 4, 8):
        assert drift(s, k) == pytest.approx(k * speed)


def test_hold_is_the_rms_over_an_outer_step():
    s, v = ramp()
    speed = float(np.sqrt((v**2).sum(-1).mean()))
    k = 4
    expected = speed * math.sqrt(sum(j * j for j in range(1, k + 1)) / k)
    assert hold(s, k) == pytest.approx(expected)


def test_extrapolation_is_exact_on_a_ramp():
    s, _ = ramp()
    for k in (1, 2, 4, 8):
        assert linear(s, k) == pytest.approx(0.0, abs=1e-9)
        assert lagged(s, k) == pytest.approx(0.0, abs=1e-9)


def test_extrapolation_needs_enough_frames():
    s, _ = ramp(frames=9)
    assert linear(s, 8) is None
    assert lagged(s, 4) is None


def test_acceptance_proxy_limits():
    flat = np.zeros((3, 17))
    stats = energy_stats(flat, 4, 300.0, 100)
    assert stats["acceptance_proxy"] == pytest.approx(1.0)
    assert stats["n_atoms_for_acceptance_0p6"] is None

    rng = np.random.default_rng(1)
    noisy = np.cumsum(rng.normal(scale=0.05, size=(3, 17)), axis=1)
    wide = energy_stats(noisy, 4, 300.0, 100)
    assert 0.0 < wide["acceptance_proxy"] < 1.0
    # a 0.1 eV spread is several kT, so only a block smaller than the cell accepts at 0.6
    assert wide["n_atoms_for_acceptance_0p6"] < 100
