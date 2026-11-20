"""Integrators and the interval controller: conservation, reduction to Verlet, temperature."""

from __future__ import annotations

import pytest
import torch
from ase import units

from frost.controller import IntervalController
from frost.integrators import Integrator, Langevin, MDState


def harmonic(kappa: float):
    """Independent 3D springs to the origin: energy, forces, unchanged positions."""

    def force(x):
        return 0.5 * kappa * (x * x).sum(), -kappa * x, x

    return force


def state(n: int = 64, seed: int = 0, temperature: float = 300.0, mass: float = 28.0) -> MDState:
    g = torch.Generator().manual_seed(seed)
    x = 0.05 * torch.randn(n, 3, generator=g, dtype=torch.float64)
    m = torch.full((n, 1), mass, dtype=torch.float64)
    v = torch.randn(n, 3, generator=g, dtype=torch.float64) * torch.sqrt(units.kB * temperature / m)
    return MDState(x, v, m)


def test_velocity_verlet_conserves_energy():
    s = state()
    fast = harmonic(10.0)
    integ = Integrator(s, dt_fs=0.5)
    e, f, _ = fast(s.x)
    e0 = float(e) + s.kinetic_energy()
    worst = 0.0
    for _ in range(2000):
        e, f = integ.inner_step(fast, f)
        worst = max(worst, abs(float(e) + s.kinetic_energy() - e0))
    assert worst / abs(e0) < 1e-3


def test_respa_with_a_zero_residual_is_plain_verlet():
    a, b = state(seed=1), state(seed=1)
    fast = harmonic(10.0)
    ia, ib = Integrator(a, dt_fs=0.5), Integrator(b, dt_fs=0.5)
    _, fa, _ = fast(a.x)
    _, fb, _ = fast(b.x)
    zero = torch.zeros_like(a.x)
    for _ in range(10):
        _, fa = ia.respa_outer(fast, fa, zero, k=4)
        ia.kick(zero, 0.5 * 4 * ia.dt)
    for _ in range(40):
        _, fb = ib.inner_step(fast, fb)
    assert torch.allclose(a.x, b.x, atol=1e-12)
    assert torch.allclose(a.v, b.v, atol=1e-12)


def test_respa_calls_back_after_every_inner_step():
    """Observables must be sampled at the inner rate, not once per outer step."""
    s = state(seed=5)
    fast = harmonic(10.0)
    integ = Integrator(s, dt_fs=0.5)
    _, f, _ = fast(s.x)
    seen = []
    integ.respa_outer(fast, f, torch.zeros_like(s.x), k=4, on_step=lambda: seen.append(s.x.clone()))
    assert len(seen) == 4
    assert not torch.allclose(seen[0], seen[-1])


def test_respa_conserves_the_energy_of_a_split_potential():
    s = state(seed=2)
    k_fast, k_slow = 40.0, 1.0
    fast, slow = harmonic(k_fast), harmonic(k_slow)
    integ = Integrator(s, dt_fs=0.5)
    _, f_fast, _ = fast(s.x)
    _, residual, _ = slow(s.x)

    def total() -> float:
        return 0.5 * (k_fast + k_slow) * float((s.x * s.x).sum()) + s.kinetic_energy()

    e0 = total()
    worst = 0.0
    for _ in range(500):
        _, f_fast = integ.respa_outer(fast, f_fast, residual, k=4)
        _, residual, _ = slow(s.x)
        integ.kick(residual, 0.5 * 4 * integ.dt)
        worst = max(worst, abs(total() - e0))
    assert worst / e0 < 5e-3


def test_baoab_holds_the_target_temperature():
    s = state(n=2000, seed=3, temperature=100.0)
    fast = harmonic(10.0)
    thermostat = Langevin(dt_fs=1.0, friction_per_fs=0.05, temperature_K=300.0,
                          masses=s.masses, seed=4)
    integ = Integrator(s, dt_fs=1.0, thermostat=thermostat)
    _, f, _ = fast(s.x)
    temperatures = []
    for step in range(3000):
        _, f = integ.inner_step(fast, f)
        if step >= 1000:
            temperatures.append(s.temperature())
    assert sum(temperatures) / len(temperatures) == pytest.approx(300.0, rel=0.02)


def test_a_budget_violation_halves_the_interval():
    c = IntervalController(epsilon=20.0, k=8)
    assert c.observe(staleness=60.0, k_used=8) == 4


def test_quiet_outer_steps_double_the_interval():
    c = IntervalController(epsilon=20.0, k=2, quiet_to_grow=3)
    for _ in range(3):
        c.observe(staleness=3.0, k_used=c.k)
    assert c.k == 4


def test_resonance_cap_admits_the_interval_that_passed_50_ps_and_refuses_the_next():
    """Calibrated on 50 ps NVE: 0.200 of the period passes, 0.242 fails."""
    c = IntervalController(epsilon=2.0, k=4, quiet_to_grow=1, period_fs=30.0, dt_fs=1.0)
    for _ in range(6):
        c.observe(staleness=0.5, k_used=c.k)
    assert c.resonance_cap() == 6  # 0.2 x 30 fs, on the even lattice
    assert c.k == 6  # never the 8 that gains 1.81 meV/atom over 50 ps

    # Li3PO4: 0.2 x 33 = 6.6 fs, and 6 is what its 50 ps run passes at
    assert IntervalController(period_fs=33.0, dt_fs=1.0).resonance_cap() == 6
    # water's 9 fs period leaves no usable interval at all
    assert IntervalController(period_fs=9.0, dt_fs=1.0).resonance_cap() == 1


def test_the_interval_lattice_is_even_so_six_is_reachable():
    """A doubling lattice cannot express silica's passing k = 6 and hands out 8."""
    c = IntervalController(epsilon=2.0, k=8, period_fs=30.0, dt_fs=1.0)
    c.observe(staleness=100.0, k_used=8)  # over budget: halve, onto the even lattice
    assert c.k == 4
    assert IntervalController().even(7) == 6
    assert IntervalController().even(2) == 2
    assert IntervalController().even(1) == 1
    # even in femtoseconds: at dt = 2 fs every step count is on the lattice
    assert IntervalController(dt_fs=2.0).even(5) == 5
    assert IntervalController(dt_fs=0.5).even(7) == 4


def test_without_a_period_only_k_max_limits_growth():
    c = IntervalController(epsilon=2.0, k=4, k_max=16, quiet_to_grow=1)
    for _ in range(6):
        c.observe(staleness=0.5, k_used=c.k)
    assert c.k == 16


def test_cost_model_matches_the_note():
    assert IntervalController(c_student=0.05).predicted_speedup(4) == pytest.approx(3.17, abs=0.01)


def test_an_expensive_student_falls_back_to_the_teacher():
    assert not IntervalController(c_student=0.6, k=2).use_student()
    assert IntervalController(c_student=0.05, k=4).use_student()


def test_jittered_outer_step_keeps_its_mean_and_respects_the_cap():
    """Breaking r-RESPA's fixed period ratio must not change the mean interval."""
    import torch

    from frost.controller import IntervalController

    controller = IntervalController(k=8, k_jitter=0.5, period_fs=30.0, dt_fs=1.0)
    gen = torch.Generator().manual_seed(0)
    draws = [controller.draw(gen) for _ in range(4000)]
    assert min(draws) == 4 and max(draws) == 12  # symmetric about k
    # the cap bounds the mean interval, not each draw: at k = cap a one-sided window
    # would quietly lower the mean, and with it the speedup being compared
    assert abs(sum(draws) / len(draws) - 8.0) < 0.15
    assert len(set(draws)) > 2  # actually varying


def test_no_jitter_leaves_the_interval_alone():
    import torch

    from frost.controller import IntervalController

    controller = IntervalController(k=8, period_fs=30.0, dt_fs=1.0)
    gen = torch.Generator().manual_seed(0)
    assert {controller.draw(gen) for _ in range(50)} == {8}


def _two_timescale(k_fast: float, k_slow: float, k: int, scheme: str, steps: int = 400):
    """Energy error of a fast+slow harmonic system under one r-RESPA form."""
    import torch

    from frost.integrators import Integrator, MDState

    masses = torch.ones(1, 1, dtype=torch.float64)
    state = MDState(torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
                    torch.zeros(1, 3, dtype=torch.float64), masses)
    integ = Integrator(state, 1.0)

    def fast(x):
        return 0.5 * k_fast * (x * x).sum(), -k_fast * x, x

    def slow_force(x):
        return -k_slow * x

    def energy(s):
        return float(0.5 * (k_fast + k_slow) * (s.x * s.x).sum() + s.kinetic_energy())

    e0 = energy(state)
    f = fast(state.x)[1]
    for _ in range(steps):
        if scheme == "midpoint":
            _, f, _ = integ.respa_midpoint(fast, f, lambda x, ff: (slow_force(x), None), k)
        else:
            integ.kick(slow_force(state.x), 0.5 * k * integ.dt)
            for _ in range(k):
                _, f = integ.inner_step(fast, f)
            integ.kick(slow_force(state.x), 0.5 * k * integ.dt)
    return abs(energy(state) - e0) / abs(e0), state


def test_midpoint_respa_conserves_energy_better_than_the_boundary_form():
    """Halving the distance between where the slow force is measured and applied."""
    boundary, _ = _two_timescale(4.0, 0.25, k=8, scheme="boundary")
    midpoint, _ = _two_timescale(4.0, 0.25, k=8, scheme="midpoint")
    assert midpoint < boundary


def test_midpoint_respa_is_time_reversible():
    import torch

    from frost.integrators import Integrator, MDState

    masses = torch.ones(1, 1, dtype=torch.float64)
    x0 = torch.tensor([[0.7, -0.2, 0.1]], dtype=torch.float64)
    v0 = torch.tensor([[0.05, 0.0, -0.02]], dtype=torch.float64)
    state = MDState(x0.clone(), v0.clone(), masses)
    integ = Integrator(state, 1.0)

    def fast(x):
        return 0.5 * 4.0 * (x * x).sum(), -4.0 * x, x

    slow_at = lambda x, ff: (-0.25 * x, None)  # noqa: E731
    f = fast(state.x)[1]
    for _ in range(20):
        _, f, _ = integ.respa_midpoint(fast, f, slow_at, 8)
    state.v = -state.v
    f = fast(state.x)[1]
    for _ in range(20):
        _, f, _ = integ.respa_midpoint(fast, f, slow_at, 8)
    assert torch.allclose(state.x, x0, atol=1e-9)
    assert torch.allclose(state.v, -v0, atol=1e-9)


def test_drift_monitor_separates_the_runs_50_ps_fidelity_separates():
    """Replays the recorded energy traces: it must admit passes and refuse failures."""
    import json
    from pathlib import Path

    from frost.controller import DriftMonitor

    results = Path(__file__).resolve().parents[2] / "data" / "results" / "e6"

    def worst(name: str) -> float:
        trace = json.loads((results / f"{name}.json").read_text())
        trace = trace["frost"]["energy_trace_meV_per_atom_1ps"]
        monitor = DriftMonitor(budget=0.02, window_ps=10.0)
        seen = [monitor.observe(e, ps * 1e3) for ps, e in enumerate(trace)]
        return max((abs(d) for d in seen if d is not None), default=0.0)

    passes = ["fidelity_silica1152_T300_nve_cheap_k6_50ps", "fidelity_li3po4_T300_nve_cheap_k6_50ps",
              "fidelity_si512_T300_nve_cheap_k8_50ps", "fidelity_si512_T300_nve_cheap_k10_50ps",
              "fidelity_cu500_T300_nve_cheap_k16_50ps"]
    passes.append("fidelity_cu500_T300_nve_cheap_k24_50ps")
    failures = ["fidelity_silica1152_T300_nve_50ps_trace", "fidelity_li3po4_T300_nve_k8_50ps",
                "fidelity_si512_T300_nve_cheap_k14_50ps",
                # the tightest case: 0.0348 against a 0.029 budget
                "fidelity_cu500_T300_nve_cheap_k32_50ps"]
    budget = DriftMonitor().budget
    for name in passes:
        if (results / f"{name}.json").exists():
            assert worst(name) <= budget, name  # Si at k = 10 is the tightest, at 0.0248
    for name in failures:
        if (results / f"{name}.json").exists():
            assert worst(name) > budget, name


def test_drift_monitor_waits_for_a_window_before_judging():
    """A short window measures fluctuation, not drift."""
    from frost.controller import DriftMonitor

    monitor = DriftMonitor(budget=0.02, window_ps=10.0)
    assert monitor.observe(0.0, 0.0) is None
    assert monitor.observe(5.0, 1e3) is None  # a huge apparent slope, too early to judge
    for ps in range(2, 8):
        monitor.observe(0.0, ps * 1e3)
    assert monitor.observe(0.0, 8e3) is not None  # half the window and eight points


def test_projected_gain_separates_the_same_runs_without_a_fitted_threshold():
    """Its threshold is G3's own criterion, 1 meV/atom over 50 ps, tuned to nothing."""
    import json
    from pathlib import Path

    from frost.controller import DriftMonitor

    results = Path(__file__).resolve().parents[2] / "data" / "results" / "e6"

    def projection(name: str) -> float | None:
        trace = json.loads((results / f"{name}.json").read_text())
        trace = trace["frost"]["energy_trace_meV_per_atom_1ps"]
        monitor = DriftMonitor()
        seen = [monitor.projected_gain(e, ps * 1e3) for ps, e in enumerate(trace[:21])]
        return next((p for p in reversed(seen) if p is not None), None)

    passes = ["fidelity_silica1152_T300_nve_cheap_k6_50ps", "fidelity_li3po4_T300_nve_cheap_k6_50ps",
              "fidelity_si512_T300_nve_cheap_k10_50ps", "fidelity_cu500_T300_nve_cheap_k24_50ps"]
    failures = ["fidelity_silica1152_T300_nve_50ps_trace", "fidelity_li3po4_T300_nve_k8_50ps",
                "fidelity_si512_T300_nve_cheap_k14_50ps"]
    budget = DriftMonitor().gain_budget
    for name in passes:
        if (results / f"{name}.json").exists():
            assert projection(name) < budget, name
    for name in failures:
        if (results / f"{name}.json").exists():
            assert projection(name) > budget, name


def test_projection_waits_for_min_ps_and_resets_with_the_interval():
    """Li3PO4 projects to 0.85 at 10 ps and 0.11 by 20: an early transient decays."""
    from frost.controller import DriftMonitor

    monitor = DriftMonitor()
    for ps in range(15):
        assert monitor.projected_gain(0.1 * ps, ps * 1e3) is None  # under min_ps
    for ps in range(15, 25):
        monitor.projected_gain(0.1 * ps, ps * 1e3)
    assert monitor.projected_gain(2.4, 25e3) is not None
    monitor.reset()  # the interval changed; the old accumulation describes another one
    assert monitor.projected_gain(0.0, 26e3) is None


def test_a_refused_interval_is_never_re_adopted():
    """Growth runs on 480 fs and refusal on 20,000; without a latch, growth always wins."""
    from frost.controller import IntervalController

    c = IntervalController(k=16, k_max=16, period_fs=78.7, dt_fs=1.0, quiet_to_grow=1)
    c.refuse(16)
    assert c.k == 8  # halves at once
    assert c.ceiling == 14  # and one rung below the failure, for good
    for _ in range(10):  # every quiet step tries to grow back
        c.observe(staleness=0.4, k_used=c.k)
    assert c.k <= 14, "the growth rule returned to a measured failure"


def test_refusals_ratchet_down_to_the_last_passing_rung():
    """Si fails at 16, 14 and 12 and passes at 10; the ceiling must converge there."""
    from frost.controller import IntervalController

    c = IntervalController(k=16, k_max=16, period_fs=78.7, dt_fs=1.0, quiet_to_grow=1)
    reached = []
    for _ in range(3):
        c.refuse(c.k if c.k > 2 else 16)
        for _ in range(4):
            c.observe(staleness=0.4, k_used=c.k)
        reached.append(c.k)
    assert reached == [14, 12, 10]
    assert c.ceiling == 10


def test_the_ceiling_only_ever_falls():
    from frost.controller import IntervalController

    c = IntervalController(k=16, k_max=16, period_fs=78.7, dt_fs=1.0)
    c.refuse(10)
    assert c.ceiling == 8
    c.refuse(16)  # a later refusal at a larger interval must not raise it
    assert c.ceiling == 8


def test_reading_the_projection_twice_in_one_step_does_not_bias_it():
    """Calling projected_gain then over_projection must not feed the sample twice."""
    from frost.controller import DriftMonitor

    once, twice = DriftMonitor(), DriftMonitor()
    for ps in range(40):
        energy = 0.02 * ps
        once.projected_gain(energy, ps * 1e3)
        twice.projected_gain(energy, ps * 1e3)
        twice.over_projection(energy, ps * 1e3)  # the same sample, read again
    assert len(once.adopted) == len(twice.adopted)
    assert once.projected_gain(0.8, 40e3) == pytest.approx(twice.projected_gain(0.8, 40e3))


def test_effective_energy_is_conserved_under_langevin():
    """E - Q (Q the heat the O steps added) stays put under BAOAB while E itself wanders with the bath."""
    s = state(seed=3)
    fast = harmonic(10.0)
    integ = Integrator(s, dt_fs=0.5, thermostat=Langevin(0.5, 0.01, 300.0, s.masses, seed=1))
    integ.track_heat()
    e, f, _ = fast(s.x)
    e0 = float(e) + s.kinetic_energy()
    worst_eff, worst_total = 0.0, 0.0
    for _ in range(2000):
        e, f = integ.inner_step(fast, f)
        total = float(e) + s.kinetic_energy()
        worst_eff = max(worst_eff, abs(total - float(integ.heat) - e0))
        worst_total = max(worst_total, abs(total - e0))
    assert worst_total > 50 * worst_eff  # the bath moves E; it cannot move E - Q
    assert worst_eff / abs(e0) < 1e-3


def test_effective_energy_sees_a_spurious_force():
    """A non-conservative error (a lagging force) shows as drift of E - Q even though the thermostat holds T."""
    s = state(seed=4)
    fast = harmonic(10.0)
    lag = {"f": None}

    def lagged(x):
        e, f, x = fast(x)
        out = lag["f"] if lag["f"] is not None else f
        lag["f"] = f
        return e, out, x

    exact, stale = [Integrator(state(seed=4), dt_fs=0.5, thermostat=Langevin(0.5, 0.01, 300.0, s.masses, seed=2))
                    for _ in range(2)]
    drift = []
    for integ, force in ((exact, fast), (stale, lagged)):
        integ.track_heat()
        e, f, _ = force(integ.state.x)
        e0 = float(e) + integ.state.kinetic_energy()
        for _ in range(2000):
            e, f = integ.inner_step(force, f)
        drift.append(abs(float(e) + integ.state.kinetic_energy() - float(integ.heat) - e0))
    assert drift[1] > 20 * drift[0]


def _ramp_run(predictive: bool, steps: int = 60000):
    """A drifting system, staleness c(t) k^1.3 with c rising tenfold over the run, driven the executor's way: an
    interval over epsilon is redone as two halves. Returns (rollbacks, intervals, mean k)."""
    ctrl = IntervalController(k=8, k_max=64, predictive=predictive)
    t, rollbacks, intervals, ks = 0, 0, 0, []
    while t < steps:
        c = 0.01 * (1.0 + 9.0 * t / steps)
        k = max(1, min(ctrl.k, steps - t))
        s = c * k**1.3 * 3**0.5  # observe() takes the change, sqrt(3) x the mean staleness
        parts = [k] if s / 3**0.5 <= ctrl.epsilon or k == 1 else [k // 2, k - k // 2]
        rollbacks += len(parts) - 1
        for part in parts:
            ctrl.observe(c * part**1.3 * 3**0.5, part)
            ks.append(part)
            intervals += 1
            t += part
    return rollbacks, intervals, sum(ks) / len(ks)


def test_predictive_controller_rolls_back_less_on_a_ramp():
    aimd = _ramp_run(False)
    pred = _ramp_run(True)
    assert pred[0] < 0.2 * aimd[0]  # far fewer redone intervals
    assert pred[2] >= 0.9 * aimd[2]  # without giving up interval length
