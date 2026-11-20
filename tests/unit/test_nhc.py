"""The Nose-Hoover chain (frost.integrators.NoseHooverChain) on harmonic oscillators: it holds the target temperature,
conserves its extended energy, and rewinds with a checkpoint."""

from __future__ import annotations

import torch
from ase import units

from frost.integrators import Integrator, MDState, NoseHooverChain


def _oscillators(n=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    k = 1.0 + 4.0 * torch.rand(n, 1, generator=g, dtype=torch.float64)  # eV/A^2, a spread of frequencies
    masses = torch.full((n, 1), 30.0, dtype=torch.float64)
    x = 0.05 * torch.randn(n, 3, generator=g, dtype=torch.float64)
    v = torch.zeros(n, 3, dtype=torch.float64)

    def force(pos):
        return (0.5 * k * pos * pos).sum(), -k * pos, pos

    return MDState(x, v, masses), force, k


def test_chain_holds_the_temperature_and_conserves_its_extended_energy():
    state, force, k = _oscillators()
    thermo = NoseHooverChain(dt_fs=1.0, tau_fs=50.0, temperature_K=300.0, masses=state.masses)
    integ = Integrator(state, 1.0, thermo)
    e, f, state.x = force(state.x)
    temps, extended = [], []
    for step in range(20000):
        e, f = integ.inner_step(force, f)
        if step > 4000 and step % 10 == 0:
            temps.append(state.temperature())
        if step % 100 == 0:
            extended.append(float(e) + state.kinetic_energy() + thermo.conserved_extra())
    t_mean = sum(temps) / len(temps)
    assert abs(t_mean - 300.0) < 15.0  # 128 oscillators: ~5% scatter of the running mean at most
    drift = max(extended) - min(extended)
    assert drift < 1e-3 * 3 * 128 * units.kB * 300.0  # the extended energy holds to 0.1% of the thermal energy


def test_chain_state_rewinds_with_a_checkpoint():
    state, force, _ = _oscillators(seed=1)
    thermo = NoseHooverChain(dt_fs=1.0, tau_fs=50.0, temperature_K=300.0, masses=state.masses)
    integ = Integrator(state, 1.0, thermo)
    e, f, state.x = force(state.x)
    for _ in range(200):
        e, f = integ.inner_step(force, f)
    saved = thermo.state(), state.x.clone(), state.v.clone(), f.clone()
    for _ in range(50):
        e, f = integ.inner_step(force, f)
    after = state.x.clone()
    thermo.restore(saved[0])
    state.x, state.v, f = saved[1].clone(), saved[2].clone(), saved[3].clone()
    for _ in range(50):
        e, f = integ.inner_step(force, f)
    assert torch.allclose(state.x, after)  # deterministic: the same 50 steps again give the same positions


def test_graphed_chain_matches_the_eager_update_on_gpu():
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    state, force, _ = _oscillators(seed=2)
    dev = "cuda"
    masses = state.masses.to(dev)
    eager = NoseHooverChain(dt_fs=1.0, tau_fs=50.0, temperature_K=300.0, masses=masses, graphed=False)
    graphed = NoseHooverChain(dt_fs=1.0, tau_fs=50.0, temperature_K=300.0, masses=masses, graphed=True)
    g = torch.Generator(device=dev).manual_seed(0)
    v = torch.randn(state.v.shape, generator=g, device=dev, dtype=torch.float64) * 0.01
    va, vb = v.clone(), v.clone()
    for step in range(200):
        va, vb = eager(va), graphed(vb)
        if step == 100:  # a temperature change and a rewind act on the captured buffers
            for th in (eager, graphed):
                th.set_temperature(500.0)
            saved = graphed.state()
            vb_saved = vb.clone()
            vb = graphed(vb)
            graphed.restore(saved)
            vb = vb_saved
    assert float((va - vb).abs().max()) < 1e-12
    assert float((eager.vxi - graphed.vxi).abs().max()) < 1e-12 and float((eager.xi - graphed.xi).abs().max()) < 1e-12
