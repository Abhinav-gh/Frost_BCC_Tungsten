"""BandGLE: canonical at any friction shape, and friction where it is asked for."""

from __future__ import annotations

import math

import pytest
import torch
from ase import units

from frost.integrators import BandGLE, Integrator, MDState

K_SPRING = 1.0  # eV/A^2
MASS = 16.0  # amu


@pytest.fixture(autouse=True)
def float64_one_thread():
    threads, dtype = torch.get_num_threads(), torch.get_default_dtype()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(dtype)
    torch.set_num_threads(threads)


def spring(k):
    def force(x):
        return torch.tensor(0.0), -k * x, x
    return force


def oscillator_thz(k: float) -> float:
    w = math.sqrt(k / MASS)  # rad per ASE time unit; one fs is units.fs ASE time units
    return w * units.fs / (2 * math.pi) * 1e3  # cycles per ps


def test_band_gle_samples_the_canonical_distribution():
    n = 400  # independent oscillators
    masses = torch.full((n, 1), MASS)
    state = MDState(torch.zeros(n, 3), torch.zeros(n, 3), masses)
    f = oscillator_thz(K_SPRING)
    gle = BandGLE(0.5, 300.0, masses, center_thz=f, width_thz=0.2 * f, peak_per_fs=0.05,
                  base_per_fs=0.001, seed=3)
    integ = Integrator(state, 0.5, gle)
    force = spring(K_SPRING)
    f_now = force(state.x)[1]
    kin, pot = [], []
    for step in range(6000):
        _, f_now = integ.inner_step(force, f_now)
        if step > 1000:
            kin.append(state.kinetic_energy() / n)
            pot.append(float(0.5 * K_SPRING * (state.x ** 2).sum()) / n)
    kt = units.kB * 300.0
    assert sum(kin) / len(kin) == pytest.approx(1.5 * kt, rel=0.03)
    assert sum(pot) / len(pot) == pytest.approx(1.5 * kt, rel=0.05)


@pytest.mark.parametrize("ratio", [1.0, 0.3])
def test_band_gle_damps_the_band_and_spares_the_rest(ratio):
    """Energy decay rate of a noiseless oscillator matches gamma(w) from the kernel."""
    f0 = oscillator_thz(K_SPRING)
    k = K_SPRING * ratio * ratio  # oscillator at ratio * f0
    masses = torch.full((1, 1), MASS)
    state = MDState(torch.tensor([[0.1, 0.0, 0.0]]), torch.zeros(1, 3), masses)
    # weak coupling: the peak (0.002 per fs) is a fifth of the band's own width Gamma
    gle = BandGLE(0.2, 1e-9, masses, center_thz=f0, width_thz=0.4 * f0, peak_per_fs=0.002)
    integ = Integrator(state, 0.2, gle)
    force = spring(k)
    f_now = force(state.x)[1]
    energy = []
    steps = 20000
    for _ in range(steps):
        _, f_now = integ.inner_step(force, f_now)
        energy.append(state.kinetic_energy() + float(0.5 * k * (state.x ** 2).sum()))
    tail = torch.tensor(energy[steps // 2:])
    head = torch.tensor(energy[: steps // 10])
    rate = math.log(float(head.mean()) / float(tail.mean())) / (0.2 * (0.75 * steps - 0.05 * steps))
    expected = gle.friction(ratio * f0)
    assert rate == pytest.approx(expected, rel=0.3, abs=2e-5)
    if ratio == 1.0:
        assert expected > 3 * gle.friction(0.3 * f0)  # the band (a wide one here), not everything


def test_a_narrow_band_cannot_damp_faster_than_its_width():
    """Strong coupling: peak 0.01 on a band of Gamma = 0.0025 per fs damps at about Gamma."""
    f0 = oscillator_thz(K_SPRING)
    masses = torch.full((1, 1), MASS)
    state = MDState(torch.tensor([[0.1, 0.0, 0.0]]), torch.zeros(1, 3), masses)
    gle = BandGLE(0.2, 1e-9, masses, center_thz=f0, width_thz=0.1 * f0, peak_per_fs=0.01)
    integ = Integrator(state, 0.2, gle)
    force = spring(K_SPRING)
    f_now = force(state.x)[1]
    energy = []
    for _ in range(20000):
        _, f_now = integ.inner_step(force, f_now)
        energy.append(state.kinetic_energy() + float(0.5 * K_SPRING * (state.x ** 2).sum()))
    tail, head = torch.tensor(energy[10000:]), torch.tensor(energy[:2000])
    rate = math.log(float(head.mean()) / float(tail.mean())) / (0.2 * 14000)
    assert rate < 0.4 * gle.friction(f0)
    assert rate == pytest.approx(gle.params["gamma_aux"], rel=0.3)
