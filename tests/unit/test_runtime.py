"""The runtime's executor, clauses, reuse typing and planner, on a fake backend (CPU, no models)."""

from __future__ import annotations

import pytest

from frost.controller import IntervalController
from frost.runtime import (
    Executor, Interval, Magnitude, NonFinite, QoSThreshold, ResonancePrior, ReuseClass, check_hold,
)
from frost.runtime import planner


class FakeBackend:
    """Staleness grows with k; a step counter; checkpoint and restore of that counter."""

    def __init__(self, stale_per_step=0.2, bad_at=None):
        self.step, self.calls = 0, []
        self.stale_per_step, self.bad_at = stale_per_step, bad_at or set()

    def time(self):
        return float(self.step)

    def steps_left(self, horizon):
        return max(0, int(horizon) - self.step)

    def checkpoint(self):
        return self.step

    def restore(self, ck):
        self.step = ck

    def finish(self):
        pass

    def interval(self, k, mode):
        self.calls.append(k)
        start = self.step
        self.step += k
        finite = not any(start <= b < self.step for b in self.bad_at) or k == 1
        return Interval(t=float(self.step), k=k, staleness=self.stale_per_step * k, finite=finite)


def test_rollback_halves_until_the_clause_admits():
    b = FakeBackend(stale_per_step=0.5)  # k = 8 -> 4.0 > 2.0, k = 4 -> 2.0 admitted
    ex = Executor(b, [Magnitude(epsilon=2.0)], k=8)
    ex.run(16)
    assert b.time() == 16
    assert all(k == 4 for _, k, _, _ in ex.log)  # every logged interval is a half
    assert sum(e["action"] == "rollback" for e in ex.events) == 2


def test_nonfinite_interval_is_redone_down_to_k1():
    b = FakeBackend(stale_per_step=0.0, bad_at={5})
    ex = Executor(b, [NonFinite()], k=8)
    ex.run(8)
    assert b.time() == 8
    assert [k for _, k, _, _ in ex.log] == [4, 1, 1, 2]  # only the half holding step 5 is split


def test_refusal_latches_a_ceiling():
    b = FakeBackend(stale_per_step=0.1)
    ctrl = IntervalController(k=16, k_max=16)
    ex = Executor(b, [QoSThreshold(threshold=1.0)], controller=ctrl)  # refuses k = 16 (1.6) for good
    ex.run(400)
    ks = [k for _, k, _, _ in ex.log]
    assert ks[0] == 16 and max(ks[1:]) < 16
    assert ctrl.ceiling is not None and ctrl.ceiling < 16


def test_resonance_prior_caps_before_anything_runs():
    b = FakeBackend(stale_per_step=0.0)
    ex = Executor(b, [ResonancePrior(period_fs=27.7, dt_fs=1.0)], controller=IntervalController(k=16, k_max=16))
    ex.run(60)
    assert max(k for _, k, _, _ in ex.log) == 6


def test_reuse_typing_refuses_intermediate_holds_of_gradient_consumed_tensors():
    check_hold(ReuseClass.GRADIENT, "output")
    check_hold(ReuseClass.OPERATOR, "intermediate")
    with pytest.raises(ValueError):
        check_hold(ReuseClass.GRADIENT, "intermediate")
    with pytest.raises(ValueError):
        check_hold(ReuseClass.VALUE, "intermediate")


def test_planner_matches_the_cost_model():
    # Cu at k = 24: teacher 21.0 ms, student 2.30 ms per step -> 6.63x sync
    best = {m: s for s, m, k in planner.plan(21.0, 2.30, {"sync": 24}) if k == 24}
    assert abs(best["sync"] - 6.63) < 0.02
    # two devices, NVT: the lag mode is teacher-bound at k = 6 when 6 * c_s < c_t
    lag = [s for s, m, k in planner.plan(36.0, 3.0, {"lag": 6}, devices=2, nve=False) if k == 6]
    assert abs(lag[0] - 6.0) < 1e-9
    assert not [m for _, m, _ in planner.plan(36.0, 3.0, {"lag": 6}, devices=2, nve=True)]


def test_lag_compensation_is_exact_for_a_linear_curvature_mismatch():
    import torch

    from frost.async_teacher import LaggedResidual

    class Done:
        def __init__(self, f):
            self.f = f

        def result(self):
            return 0.0, self.f

    class Teacher:  # F_T = -w2 x; the student is 20% too soft
        def submit(self, x):
            return Done(-4.0 * x)

    w2, mu = 4.0, 0.2
    f_s = lambda x: -(1.0 - mu) * w2 * x  # noqa: E731
    g = torch.Generator().manual_seed(0)
    x = torch.randn(64, 3, generator=g, dtype=torch.float64)
    lag = LaggedResidual(Teacher(), Teacher().submit(x).f - f_s(x), compensate=True, f_student0=f_s(x))
    for _ in range(6):
        x = x + 0.1 * torch.randn(64, 3, generator=g, dtype=torch.float64)
        kick, _ = lag.boundary(x, f_s(x), 0.0, 0)
    exact = Teacher().submit(x).f - f_s(x)
    assert torch.allclose(kick, exact, atol=1e-9)  # R(x_{n-1}) + mu_hat dF_S == R(x_n)
    plain = lag.history[-1]
    assert (plain - exact).abs().max() > 1e-2  # the uncompensated lag is not


def test_student_choice_trades_accuracy_against_cost():
    # a cheap student that only holds k = 4 against a dearer one that holds k = 16 (cap 16)
    replays = {"cheap": {"hold_R": {"4": 0.010, "8": 0.030, "16": 0.08}, "student_ms": 0.5},
               "accurate": {"hold_R": {"4": 0.002, "8": 0.004, "16": 0.008}, "student_ms": 2.0}}
    ranked = planner.choose_student(replays, rms_inv_mass=0.05, resonance_cap=16, epsilon=2.0, teacher_ms=20.0)
    best = {name: (s, k) for s, name, k in ranked}
    assert best["accurate"][1] == 16 and best["cheap"][1] == 9  # 1.5 at k = 8, 4.0 at 16: eps 2 interpolates to 9
    # cheap: 1 / (0.9 / 20 + 1 / 9) = 6.41; accurate: 1 / (2.4 / 20 + 1 / 16) = 5.48 -- the cheapest adequate student wins
    assert ranked[0][1] == "cheap" and abs(ranked[0][0] - 6.41) < 0.01 and abs(best["accurate"][0] - 5.48) < 0.01


def test_canary_refuses_a_biased_schedule_and_promotes_an_unbiased_one():
    import numpy as np

    from frost.runtime.canary import Canary

    class Members:  # 8 members whose statistic fluctuates 5% around a level
        def __init__(self, level, seed):
            self.level, self.rng, self.step, self.u = level, np.random.default_rng(seed), 0, None

        def time(self):
            return float(self.step)

    class Ex:
        def __init__(self, b):
            self.b = b

        def run(self, until):
            while self.b.step < until:
                self.b.step += 1
            self.b.u = self.b.level * (1.0 + 0.05 * self.b.rng.standard_normal((8, 1)))

    def make(level, seed):
        b = Members(level, seed)
        return b, Ex(b)

    stats = lambda u: u  # noqa: E731
    biased = Canary(make(1.0, 0), make(1.15, 1), stats, ["energy"], burn_in=0, tolerance={"energy": 0.05})
    fair = Canary(make(1.0, 2), make(1.0, 3), stats, ["energy"], burn_in=0, tolerance={"energy": 0.05})
    for t in range(1, 201):
        biased.advance(t, 1)
        fair.advance(t, 1)
    assert biased.verdict()["verdict"] == "refuse"
    assert fair.verdict()["verdict"] == "promote"


def test_rdf_is_the_same_for_wrapped_and_unwrapped_coordinates():
    import numpy as np
    import torch

    from frost.observables import rdf

    a, n = 3.61, 4  # fcc Cu, 4 x 4 x 4 cells
    base = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    x = np.array([(b + [i, j, k]) * a for i in range(n) for j in range(n) for k in range(n) for b in base])
    cell = np.eye(3) * a * n
    rng = np.random.default_rng(0)
    x = x + 0.05 * rng.standard_normal(x.shape)
    shifted = x + rng.integers(-2, 3, size=x.shape) * a * n  # the same configuration, atoms in other images
    g1 = np.asarray(rdf(torch.as_tensor(x[None]), cell, np.zeros(len(x), int), 1, r_max=6.0)["g_total"])
    g2 = np.asarray(rdf(torch.as_tensor(shifted[None]), cell, np.zeros(len(x), int), 1, r_max=6.0)["g_total"])
    assert np.allclose(g1, g2)


class BaselineBackend(FakeBackend):
    """A cheap path whose staleness is too large at every interval, and a baseline that runs n steps."""

    def __init__(self, stale_per_step=3.0, probe_stale=None):
        super().__init__(stale_per_step=stale_per_step)
        self.baseline_calls, self.probe_stale = [], probe_stale

    def baseline_interval(self, n, probe_k, mode):
        self.baseline_calls.append(n)
        self.step += n
        stale = (self.probe_stale if self.probe_stale is not None else self.stale_per_step) * probe_k
        return Interval(t=float(self.step), k=1, staleness=stale, finite=True, extra={"probed": bool(probe_k)})


def test_never_lose_runs_the_baseline_when_the_smallest_interval_fails():
    b = BaselineBackend(stale_per_step=3.0)  # k = 2 holds 6.0 > 2.0: nothing multirate is admitted
    ex = Executor(b, [Magnitude(epsilon=2.0)], controller=IntervalController(k=4, k_max=4), never_lose=True,
                  probe_every=100, probe_every_max=800)
    res = ex.run(5000)
    assert b.time() == 5000
    assert res["baseline_steps"] >= 5000 - 6  # everything after the first failed intervals is the baseline
    assert any(e["action"] == "baseline" for e in ex.events)
    assert not any(e["action"] == "resume" for e in ex.events)
    assert max(b.baseline_calls) == 800  # failed probes back off to the cap


def test_never_lose_resumes_after_clean_probes():
    b = BaselineBackend(stale_per_step=3.0, probe_stale=0.1)  # the trouble was transient: probes are clean
    ctrl = IntervalController(k=2, k_max=4)
    ex = Executor(b, [Magnitude(epsilon=2.0)], controller=ctrl, never_lose=True, probe_every=50)
    ex.run(400)
    assert any(e["action"] == "resume" for e in ex.events)
    assert len(b.baseline_calls) >= ctrl.quiet_to_grow


def test_never_lose_is_off_for_a_fixed_interval():
    b = BaselineBackend(stale_per_step=3.0)
    ex = Executor(b, [Magnitude(epsilon=2.0)], k=4, never_lose=True)
    ex.run(40)
    assert not b.baseline_calls


def test_thermostat_bias_predicts_the_water_shift_and_refuses_only_past_tolerance():
    from frost.runtime import Interval, ThermostatBias

    def trace(clause, slope_meV_per_ps, noise=0.0, n=200):
        import random

        rng = random.Random(0)
        out = None
        for i in range(n):
            t_fs = 100.0 * (i + 1)  # 20 ps
            e = slope_meV_per_ps * t_fs / 1e3 + noise * rng.gauss(0.0, 1.0)
            out = clause.check(Interval(t=t_fs, k=4, staleness=0.0, finite=True, energy=e)) or out
        return out

    water = ThermostatBias(friction_per_ps=1.0, temperature_K=300.0, tolerance=0.005)
    assert trace(water, 0.85) is not None  # water fixed 4: +3.3 K predicted, over 1.5 K
    b = water.bias()
    assert b is not None and abs(b[0] - 0.85e-3 / (3 * 8.617333262e-5)) < 0.05
    assert trace(ThermostatBias(friction_per_ps=1.0), 0.1) is None  # 0.39 K, under 1.5 K
    hot = ThermostatBias(friction_per_ps=10.0, temperature_K=3500.0)
    assert trace(hot, 3.9) is None  # the 3500 K ramp at 10/ps: 1.5 K, far under 17.5 K
    noisy = ThermostatBias(friction_per_ps=1.0)
    assert trace(noisy, 0.0, noise=0.5) is None  # pure noise does not refuse


def test_thermostat_bias_waits_for_its_minimum_span():
    from frost.runtime import Interval, ThermostatBias

    c = ThermostatBias(friction_per_ps=1.0, min_ps=5.0)
    for i in range(40):  # 4 ps of a steep slope: too short to judge
        assert c.check(Interval(t=100.0 * (i + 1), k=4, staleness=0.0, finite=True, energy=5.0 * i)) is None


def test_kinetic_bias_refuses_a_hot_interval_and_not_scatter():
    import random

    from frost.runtime import Interval, KineticBias

    def run(clause, offset, scatter=1.5, n=200):
        rng = random.Random(1)
        out = None
        for i in range(n):  # 20 ps of 100 fs intervals
            t = 300.0 + offset + scatter * rng.gauss(0.0, 1.0)
            iv = Interval(t=100.0 * (i + 1), k=8, staleness=0.0, finite=True, extra={"T_mean": t})
            out = clause.check(iv) or out
        return out

    assert run(KineticBias(temperature_K=300.0, tolerance=0.005), 3.0) is not None  # +3 K against 1.5 K
    assert run(KineticBias(temperature_K=300.0, tolerance=0.005), 0.5) is None  # inside tolerance
    assert run(KineticBias(temperature_K=300.0, tolerance=0.005), 0.0, scatter=6.0) is None  # scatter alone
    assert run(KineticBias(temperature_K=300.0, tolerance=0.005), -3.0) is not None  # too cold refuses too


def test_probation_holds_growth_until_every_measured_clause_admits():
    ctrl = IntervalController(k=10, k_max=40, period_fs=52.49, dt_fs=1.0, resonance_fraction=0.45)
    ctrl.probation, ctrl.admitters = 10, ("thermostat-bias", "kinetic-bias")
    for _ in range(20):
        ctrl.observe(0.1, ctrl.k)
    assert ctrl.k == 10  # quiet, but probation binds at the validated prior
    ctrl.admit(10, "thermostat-bias")
    assert ctrl.probation == 10  # one admitter is not enough
    ctrl.admit(10, "kinetic-bias")
    assert ctrl.probation == 12  # one rung: max(10 + 2, even(ceil(12.5)))
    for _ in range(20):
        ctrl.observe(0.1, ctrl.k)
    assert ctrl.k == 12
    ctrl.admit(8, "thermostat-bias")  # an admission below the boundary changes nothing
    assert ctrl.probation == 12


def test_no_probation_means_the_old_growth():
    ctrl = IntervalController(k=10, k_max=40, period_fs=52.49, dt_fs=1.0, resonance_fraction=0.45)
    for _ in range(20):
        ctrl.observe(0.1, ctrl.k)
    assert ctrl.k == 22  # straight to the search bound, as before


def test_bias_admission_needs_data_at_the_interval_itself():
    from frost.runtime import Interval, ThermostatBias

    calls = []
    c = ThermostatBias(friction_per_ps=1.0, min_ps=5.0, on_admit=lambda k, name: calls.append(k))
    t = 0.0
    for _ in range(60):  # 6 ps flat at k 10: admitted
        t += 100.0
        c.check(Interval(t=t, k=10, staleness=0.0, finite=True, energy=0.0))
    assert 10 in calls
    calls.clear()
    for _ in range(30):  # 3 ps at k 12: under min_ps of its own data, not admitted (the window still holds k 10)
        t += 120.0
        c.check(Interval(t=t, k=12, staleness=0.0, finite=True, energy=0.0))
    assert 12 not in calls


def test_linear_growth_takes_a_small_step_the_halving_rule_refuses():
    def settle(linear):
        ctrl = IntervalController(k=6, k_max=14, period_fs=33.0, dt_fs=1.0, resonance_fraction=0.45)
        ctrl.probation, ctrl.linear_growth = 8, linear
        for _ in range(20):
            ctrl.observe(1.19 * 3 ** 0.5, ctrl.k)  # staleness 1.19 at k 6 (observe takes the change, sqrt 3 x)
        return ctrl.k

    # the default rule is now the doubling rule written for the step taken (2026-11-14): 1.19 x 8 / 6 = 1.59 under
    # epsilon 2.0, so it takes the step too (the old epsilon / 2 form refused it); probation's linear rule adds a 0.9 margin
    assert settle(False) == 8
    assert settle(True) == 8  # 1.19 x 8 / 6 = 1.59 under 0.9 x 2.0


def test_running_fit_matches_the_full_refit_and_is_constant_cost():
    import random
    import time

    from frost.controller import DriftMonitor
    from frost.runtime.clauses import RunningFit

    rng = random.Random(3)
    rf, pts = RunningFit(), []
    for i in range(3000):
        t, e = 0.002 * i, 0.3 * 0.002 * i + rng.gauss(0.0, 0.05)
        rf.append(t, e)
        rf.trim(2.0)
        DriftMonitor._append(pts, t, e)
        while len(pts) > 2 and pts[-1][0] - pts[0][0] > 2.0:
            pts.pop(0)
    a, b = rf.fit(), DriftMonitor._fit(pts)
    assert abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9
    t0 = time.perf_counter()
    for i in range(3000, 13000):
        rf.append(0.002 * i, 0.0)
        rf.trim(2.0)
        rf.fit()
    assert time.perf_counter() - t0 < 0.5  # 10,000 points with a 1,000-point window: constant cost each


def test_structure_clause_admits_a_matching_interval_and_refuses_a_moved_one():
    import numpy as np

    from frost.runtime import Interval, StructureClause

    rng = np.random.default_rng(0)
    base = np.exp(-0.5 * ((np.arange(90) - 30) / 4.0) ** 2) + 0.2 * np.exp(-0.5 * ((np.arange(90) - 55) / 6.0) ** 2)
    base /= base.sum()

    def frame(shift=0.0):
        h = np.roll(base, int(shift)) * (1.0 + 0.05 * rng.standard_normal(90))
        return np.clip(h, 0, None) / np.clip(h, 0, None).sum()

    admitted, c = [], None
    c = StructureClause(k_ref=6, warm_ps=0.0, ref_ps=4.0, test_ps=2.0, on_admit=lambda k, n: admitted.append(k))
    t = 0.0
    for _ in range(100):  # 6 ps at the prior: the reference is built and the prior admitted
        t += 60.0
        assert c.check(Interval(t=t, k=6, staleness=0.0, finite=True, extra={"pair_hist": frame()})) is None
    assert c.spread is not None and 6 in admitted
    for _ in range(50):  # 4 ps at k 8, unchanged structure: admitted, never refused
        t += 80.0
        assert c.check(Interval(t=t, k=8, staleness=0.0, finite=True, extra={"pair_hist": frame()})) is None
    assert 8 in admitted
    refused = None
    for _ in range(50):  # k 10 with the first peak moved two bins: refused
        t += 100.0
        refused = c.check(Interval(t=t, k=10, staleness=0.0, finite=True, extra={"pair_hist": frame(2)})) or refused
    assert refused is not None and 10 not in admitted
    c.reset()
    assert c.spread is not None  # a refusal keeps the reference


def _drift_run(monitor, slopes, noise=0.02, seed=1):
    """Energy rising at slopes[0], then slopes[1] after the first refusal; returns (refusal times, gain at 50 ps)."""
    import random

    random.seed(seed)
    e, t, refusals, slope = 0.0, 0.0, [], slopes[0]
    while t < 50.0:
        t += 0.01
        e += slope * 0.01
        if monitor.over_projection(e + random.gauss(0.0, noise), 1000.0 * t):
            refusals.append(t)
            monitor.reset()
            slope = slopes[1] if len(refusals) == 1 else 0.0
    return refusals, e


def test_sequential_drift_refuses_a_clear_failure_early_and_keeps_the_fixed_rule_otherwise():
    from frost.controller import DriftMonitor

    fixed, _ = _drift_run(DriftMonitor(), (0.04, 0.0))
    early, _ = _drift_run(DriftMonitor(sequential=True), (0.04, 0.0))
    assert 20.0 <= fixed[0] < 21.0  # the fixed rule waits for min_ps
    assert 5.0 <= early[0] < 10.0  # the CUSUM from five 1 ps blocks, the slope test from eight (7 ps)
    quiet, gain = _drift_run(DriftMonitor(sequential=True), (0.005, 0.0))
    assert quiet == [] and gain < 0.3  # a drift well inside the budget is never refused


def test_drift_ledger_carries_spent_energy_across_a_refusal():
    from frost.controller import DriftMonitor

    # the fixed rule forgets the 0.8 meV/atom spent before its refusal and admits 0.012/ps, which projects to 0.6
    # on its own: the run ends over G3's 1 meV/atom; the ledger allows the window's remainder only
    _, gain_fixed = _drift_run(DriftMonitor(), (0.04, 0.012))
    _, gain_seq = _drift_run(DriftMonitor(sequential=True), (0.04, 0.012))
    assert gain_fixed > 1.0 > gain_seq
    m = DriftMonitor(sequential=True)
    for i in range(3000):  # 30 ps at a steady 0.05/ps, then a reset: the ledger survives it, the interval does not
        m.over_projection(0.05 * i * 0.01, 10.0 * i)
    m.reset()
    assert m.blocks == [] and len(m.ledger) >= 28
    m.restart()
    assert m.ledger == []


def test_never_lose_does_not_resume_below_a_refused_ceiling():
    b = BaselineBackend(stale_per_step=3.0, probe_stale=0.1)  # clean probes would resume
    ctrl = IntervalController(k=2, k_max=4)
    ctrl.refuse(2)  # a measured clause refused the smallest interval: ceiling 1
    assert ctrl.ceiling == 1
    ex = Executor(b, [Magnitude(epsilon=2.0)], controller=ctrl, never_lose=True, probe_every=50)
    ex.run(1000)
    assert not any(e["action"] == "resume" for e in ex.events)
    assert b.calls == []  # no multirate interval at all, k 1 included: it costs more than the baseline


def test_drift_record_survives_a_never_lose_resume_only_in_sequential_mode():
    from frost.controller import DriftMonitor
    from frost.runtime import DriftPosterior

    for sequential in (False, True):
        c = DriftPosterior(monitor=DriftMonitor(sequential=sequential))
        for i in range(1200):  # 12 ps of points every 10 fs
            c.monitor.over_projection(0.001 * i, 10.0 * i)
        c.resume()
        kept = c.monitor.blocks if sequential else c.monitor.adopted
        assert (len(kept) > 0) == sequential
        c.reset()  # a refusal starts a new record either way; the ledger stays
        assert c.monitor.blocks == [] and c.monitor.adopted == []
        assert (len(c.monitor.ledger) > 0) == sequential


def test_nve_probation_after_a_drift_refusal_holds_growth_until_the_drift_clause_admits():
    from types import SimpleNamespace

    from frost.controller import DriftMonitor
    from frost.runtime import DriftPosterior

    ctrl = IntervalController(k=10, k_max=10, probation_on_refusal=True, admitters=("drift",))
    ctrl.refuse(10)
    assert (ctrl.k, ctrl.ceiling, ctrl.probation) == (4, 8, 4)
    for _ in range(50):  # quiet intervals: the old rule would be back at the ceiling (8) after five
        ctrl.observe(0.01, ctrl.k)
    assert ctrl.k == 4
    clause = DriftPosterior(monitor=DriftMonitor(sequential=True), on_admit=ctrl.admit)
    admitted = []
    for i in range(1, 1300):  # 13 ps at k 4 with a flat energy: admitted once eight blocks of its own exist
        before = ctrl.probation
        clause.check(SimpleNamespace(energy=0.001 * ((i * 7) % 5), t=10.0 * i, k=4))
        if ctrl.probation != before:
            admitted.append(i * 10.0 / 1e3)
    assert admitted and 8.0 <= admitted[0] <= 11.0 and ctrl.probation == 6
    for _ in range(10):
        ctrl.observe(0.01, ctrl.k)
    assert ctrl.k == 6  # one rung, not straight back to the ceiling


def test_drift_admission_needs_the_slope_well_inside_the_budget():
    from frost.controller import DriftMonitor

    for slope, expect in ((0.0, True), (0.015, False)):  # the budget's rate is 0.02/ps; admission wants < 0.01
        m = DriftMonitor(sequential=True)
        for i in range(1500):
            m.over_projection(slope * i * 0.01 + 0.002 * ((i * 7) % 5), 10.0 * i)
        assert m.admits(0.0) == expect


def test_sequential_drift_catches_a_late_rise_that_one_fitted_line_dilutes():
    from frost.controller import DriftMonitor

    import random

    def run(cusum_z, cusum_floor):
        rng = random.Random(0)
        m = DriftMonitor(sequential=True, cusum_z=cusum_z, cusum_floor=cusum_floor)
        for i in range(5000):  # flat (with scatter) for 25 ps, then 0.08 meV/atom/ps
            t = i * 0.01
            e = rng.gauss(0.0, 0.05) + (0.08 * (t - 25.0) if t > 25.0 else 0.0)
            if m.over_projection(e, 1000.0 * t):
                return t
        return None

    late = run(8.0, 0.05)
    assert late is not None and 25.0 < late < 30.0
    without = run(1e9, 1e9)  # the slope test alone (the CUSUM switched off)
    assert without is None or without > late + 3.0


class EnergyBackend(FakeBackend):
    """Energy rises 0.3 meV/atom/ps at k >= 8 and stays flat below; checkpoint and restore carry the energy too."""

    def __init__(self):
        super().__init__(stale_per_step=0.01)
        self.e = 0.0

    def checkpoint(self):
        return (self.step, self.e)

    def restore(self, ck):
        self.step, self.e = ck

    def interval(self, k, mode):
        self.calls.append(k)
        self.step += k
        if k >= 8:
            self.e += 0.3e-3 * k
        return Interval(t=float(self.step), k=k, staleness=0.01 * k, finite=True, energy=self.e)


def test_drift_rewind_takes_the_refused_intervals_energy_out_of_the_trajectory():
    from frost.controller import DriftMonitor
    from frost.runtime import DriftPosterior

    finals = {}
    for rewind in (False, True):
        b = EnergyBackend()
        clause = DriftPosterior(monitor=DriftMonitor(sequential=True), rewind=rewind)
        ex = Executor(b, [clause], controller=IntervalController(k=16, k_max=16))
        ex.run(30000)
        finals[rewind] = (b.e, [e["action"] for e in ex.events])
        assert b.time() == 30000
    spent, actions = finals[False]
    assert spent > 1.0 and "rewind" not in actions  # the burst at k 16 stays in the trajectory
    kept, actions = finals[True]
    assert "rewind" in actions and kept < 0.1  # rewound: the trajectory never carries it


def test_measured_step_down_drops_by_the_excess_not_by_half():
    from frost.controller import DriftMonitor

    ctrl = IntervalController(k=36, k_max=64)
    m = DriftMonitor(sequential=True)
    m.last_rates = (0.03, 0.02)  # AgI-like: 1.5x over the allowed rate
    ctrl.refuse(36, m.step_factor())
    assert 26 <= ctrl.k <= 34  # (2/3)^(1/1.5) = 0.76 of 36, not 18
    ctrl2 = IntervalController(k=26, k_max=64)
    m.last_rates = (0.35, 0.02)  # beta-Ti-like burst: 17x over: the drop is bounded at half
    ctrl2.refuse(26, m.step_factor())
    assert ctrl2.k == 12
    ctrl3 = IntervalController(k=36, k_max=64)
    ctrl3.refuse(36)  # no factor: the old halving
    assert ctrl3.k == 18


def test_fine_lattice_allows_k3_at_the_small_end_and_keeps_the_even_lattice_above():
    water = IntervalController(k_max=64, period_fs=16.78, dt_fs=1.0, resonance_fraction=0.22, fine_fs=4.0)
    assert water.resonance_cap() == 3  # 3.69 fs: k 3, where the even lattice gives 2
    assert IntervalController(k_max=64, period_fs=16.78, dt_fs=1.0, resonance_fraction=0.22).resonance_cap() == 2
    assert water.even(3) == 3 and water.even(7) == 6  # fine below 4 fs, even above
    li = IntervalController(k=6, k_max=64, fine_fs=4.0)
    li.refuse(6)
    assert li.k == 3 and li.ceiling == 4  # halving 6 now lands on 3, not 2; one rung below 6 is still 4
    li.refuse(4)
    assert li.ceiling == 3  # one rung below 4 is 3 on the fine lattice


def test_thermostat_bias_holds_the_volume_out_under_a_barostat():
    """NPT: E - Q follows ln V (the residual's unbooked share of each box move), not time. Regressed out, a volume swing
    does not refuse, and a real drift under the same swing still does (gaps.md NPT results, Cu 500: ~1,000 meV/atom per
    unit ln V, 5 ps slopes of +-0.5 meV/atom/ps without the covariate)."""
    import math
    import random

    from frost.runtime import Interval, ThermostatBias
    from frost.runtime.clauses import RunningFit, RunningFitCov

    def record(slope, n=300, coupling=1000.0, seed=0):
        rng = random.Random(seed)
        lnv, out = math.log(6040.0), []
        for i in range(n):  # 100 fs points, 30 ps; ln V an OU process with ~2.4e-3 spread and ~1 ps memory
            lnv += -0.1 * (lnv - math.log(6040.0)) + 2.4e-3 * math.sqrt(2 * 0.1) * rng.gauss(0, 1)
            t_fs = 100.0 * (i + 1)
            out.append((t_fs, slope * t_fs / 1e3 + coupling * (lnv - math.log(6040.0)) + 0.02 * rng.gauss(0, 1), lnv))
        return out

    plain, cov = RunningFit(), RunningFitCov()
    for t, e, u in record(0.0)[:60]:  # one 6 ps window
        plain.append(t / 1e3, e)
        cov.append(t / 1e3, e, u)
    assert abs(cov.fit()[0]) < 3 * cov.fit()[1] + 0.01  # with ln V held out: no slope
    assert abs(cov.fit()[0]) < abs(plain.fit()[0])  # the time-only fit reads the volume swing as a slope

    def run(slope, seed):
        c = ThermostatBias(friction_per_ps=1.0, temperature_K=300.0, tolerance=0.005)
        out = None
        for t, e, u in record(slope, seed=seed):
            out = c.check(Interval(t=t, k=24, staleness=0.0, finite=True, energy=e, extra={"volume": math.exp(u)})) or out
        return out

    assert all(run(0.0, s) is None for s in range(5))  # no drift: never refused, whatever the volume does
    assert run(1.0, 0) is not None  # 1 meV/atom/ps (+3.9 K predicted) under the same swing: refused


def test_drift_budget_follows_the_run_length():
    """A steady 0.008 meV/atom/ps (the 1 ns Li3PO4 run's rate, +8 meV/atom over 1 ns) passes G3's per-50-ps budget
    (0.02/ps) but is refused when the run's own length is the horizon (2 meV/atom over 1,000 ps: 0.002/ps)."""
    import random

    from frost.controller import DriftMonitor

    def first_refusal(monitor, slope=0.008, ps=60.0):
        rng = random.Random(3)
        t = 0.0
        while t < ps:
            t += 0.01
            if monitor.over_projection(slope * t + rng.gauss(0.0, 0.02), 1000.0 * t):
                return t
        return None

    assert first_refusal(DriftMonitor(sequential=True)) is None
    t = first_refusal(DriftMonitor(sequential=True, horizon_ps=1000.0, gain_budget=2.0))
    assert t is not None and t < 40.0


def test_growth_takes_a_short_step_that_stays_under_epsilon():
    """The default growth rule scaled to the step: at k 2 with staleness 3.76 under epsilon 6.32 and a cap of 3 (the
    fine lattice), growth to 3 is predicted at 5.6 < 6.32 and taken; the doubling rule's epsilon / 2 (3.16) refused it
    (the peptide in NVE, 2026-11-14). A doubling still needs ewma < epsilon / 2, as before."""
    import math

    from frost.controller import IntervalController

    c = IntervalController(k=2, k_max=16, epsilon=6.32, period_fs=14.59, dt_fs=1.0, resonance_fraction=0.22, fine_fs=4.0)
    assert c.resonance_cap() == 3
    for _ in range(c.quiet_to_grow + 1):
        c.observe(3.76 * math.sqrt(3.0), 2)  # observe takes the change; it divides by sqrt(3) itself
    assert c.k == 3
    d = IntervalController(k=4, k_max=64, epsilon=2.0)  # no cap: k_next = 8, a doubling
    for _ in range(d.quiet_to_grow + 1):
        d.observe(1.1 * math.sqrt(3.0), 4)  # ewma 1.1 > epsilon / 2: the doubling is refused, as before
    assert d.k == 4


def test_resonance_prior_uses_the_controllers_fine_lattice():
    # the peptide in NVE (2026-11-14): 0.22 x 14.59 fs = 3.2 fs. The controller's fine lattice (every step count below
    # 4 fs) allows k 3; the prior clause rebuilt the cap on the even lattice and the executor capped k at 2, silently
    from frost.controller import IntervalController
    from frost.runtime import Executor, ResonancePrior

    ctrl = IntervalController(k=3, k_max=16, period_fs=14.59, dt_fs=1.0, fine_fs=4.0)
    assert ctrl.resonance_cap() == 3
    prior = ResonancePrior(period_fs=14.59, dt_fs=1.0, fraction=0.22)
    assert prior.cap({}) == 2  # the even lattice alone: 3.2 fs -> 2 fs
    ex = Executor(object(), [prior], controller=ctrl)
    assert ex.cap == 3
    assert ex._k_now(100) == 3 and ex.cap_bound == 0
    ctrl.k = 4
    assert ex._k_now(100) == 3 and ex.cap_bound == 1  # the bind is counted
    assert ResonancePrior(period_fs=14.59, dt_fs=1.0, fraction=0.22, fine_fs=0.0).cap({"fine_fs": 4.0}) == 2


def test_predictive_admission_steps_as_far_as_the_margin_allows():
    from frost.controller import IntervalController
    from frost.runtime import Interval, ThermostatBias

    def ctrl(admitters):
        c = IntervalController(k=24, k_max=96, period_fs=113.7, dt_fs=1.0, resonance_fraction=0.45)
        c.probation, c.admitters = 24, admitters
        return c

    c = ctrl(("thermostat-bias",))
    c.admit(24, "thermostat-bias")
    assert c.probation == 30  # the one-rung climb: even(ceil(1.25 x 24))
    c = ctrl(("thermostat-bias",))
    c.admit(24, "thermostat-bias", 2.0)
    assert c.probation == 48
    c = ctrl(("thermostat-bias", "structure"))  # an admitter without a margin keeps the one-rung climb
    c.admit(24, "thermostat-bias", 2.0)
    c.admit(24, "structure")
    assert c.probation == 30

    # the clause: a flat effective energy (tiny bias) admits with the full x2; a bias near the bar with the default x1.25
    def feed(slope_meV_per_ps):
        calls = []
        tb = ThermostatBias(friction_per_ps=1.0, temperature_K=300.0, tolerance=0.005, predictive=True,
                            on_admit=lambda k, name, g=None: calls.append(g))
        for i in range(300):
            t = 24.0 * i
            wiggle = 0.002 * ((i * 7919) % 13 - 6)  # deterministic scatter, meV/atom
            tb.check(Interval(t=t, k=24, staleness=1.0, energy=slope_meV_per_ps * t / 1e3 + wiggle))
        return calls

    calls = feed(0.0)
    assert calls and calls[-1] == 2.0
    # 0.5 x tolerance is 0.75 K, i.e. 0.194 meV/atom/ps at 1/ps: a bias at 60% of that bar steps only x1.25
    calls = feed(0.6 * 0.194)
    assert calls and abs(calls[-1] - 1.25) < 1e-12


def test_probation_never_starts_below_the_smallest_paying_interval():
    # a prior of one step (the peptide without heavy hydrogens) would deadlock: k 1 runs as the baseline, which no
    # measured clause judges, so nothing could ever admit past it, and growth pulled each resumed k 2 back to 1
    from frost.controller import IntervalController

    c = IntervalController(k=2, k_max=16, period_fs=8.4, dt_fs=1.0, resonance_fraction=0.45, fine_fs=4.0)
    prior = IntervalController(k_max=16, period_fs=8.4, dt_fs=1.0, resonance_fraction=0.22, fine_fs=4.0).resonance_cap()
    assert prior == 1
    assert c.start_probation(prior) == 2
    c.linear_growth = True
    for _ in range(20):  # quiet intervals at k 2: growth may not pull k under the probation floor
        c.observe(0.3 * c.epsilon * 3 ** 0.5, 2)
    assert c.k == 2
    assert IntervalController(period_fs=113.7, dt_fs=1.0).start_probation(24) == 24  # a prior above the floor stays


class RareFailBackend(BaselineBackend):
    """The smallest interval passes except one attempt in ``every``."""

    def __init__(self, every):
        super().__init__(stale_per_step=0.5, probe_stale=0.5)
        self.every, self.attempts = every, 0

    def interval(self, k, mode):
        iv = super().interval(k, mode)
        self.attempts += 1
        if self.attempts % self.every == 0:
            iv.staleness = 10.0
        return iv


def test_never_lose_falls_back_only_when_failures_stop_the_cheap_path_paying():
    # one failure in 200 attempts at k 2: each is redone on the baseline, and the run never latches
    b = RareFailBackend(every=200)
    ctrl = IntervalController(k=2, k_max=2)
    ex = Executor(b, [Magnitude(epsilon=2.0)], controller=ctrl, never_lose=True)
    res = ex.run(20000)
    fails = sum(e["action"] == "rollback" for e in ex.events)
    assert fails >= 40
    assert not any(e["action"] == "baseline" for e in ex.events)
    assert res["baseline_steps"] == 2 * fails  # only the failed intervals, redone exactly
    assert abs(ex.small_limit() - 0.5 * (1.01 - 0.57) / 1.01) < 1e-12
    # every attempt failing: the rate passes the limit within ~12 failures and the run latches, as before
    b = BaselineBackend(stale_per_step=3.0)
    ex = Executor(b, [Magnitude(epsilon=2.0)], controller=IntervalController(k=2, k_max=2), never_lose=True)
    ex.run(2000)
    first = next(i for i, e in enumerate(ex.events) if e["action"] == "baseline")
    assert 8 <= sum(e["action"] == "rollback" for e in ex.events[:first]) <= 16


def test_certified_store_round_trip_and_warm_start(tmp_path):
    from frost.controller import IntervalController
    from frost.runtime.store import config_key, load, save, warm_start

    cfg = {"run": "cu500_T300", "epsilon": 2.0, "period": 113.7, "bias": 0.005}
    key = config_key(cfg)
    assert key == config_key(dict(reversed(list(cfg.items()))))  # order does not matter
    assert key != config_key(dict(cfg, epsilon=2.1))  # any certificate setting changes the key
    path = tmp_path / "store.json"
    assert load(path, key) is None
    save(path, key, {"k": 50, "probation": 50, "ceiling": None, "tag": "nhc3pa_s0"})
    rec = load(path, key)
    assert rec["k"] == 50 and rec["tag"] == "nhc3pa_s0" and "saved" in rec
    c = IntervalController(k=2, k_max=96, period_fs=113.7, dt_fs=1.0, resonance_fraction=0.45)
    c.start_probation(24)
    applied = warm_start(c, rec)
    assert c.probation == 50 and c.k == 50 and applied["k"] == 50
    c2 = IntervalController(k=2, k_max=40, period_fs=113.7, dt_fs=1.0, resonance_fraction=0.45)
    c2.start_probation(24)
    warm_start(c2, rec)  # this run's own bounds still apply: k_max 40
    assert c2.k <= 40 and c2.probation <= 40


def test_small_n_allowance_scales_the_per_interval_threshold_only():
    from frost.runtime import Interval, Magnitude

    plain = Magnitude(epsilon=4.0)
    at_ref = Magnitude(epsilon=4.0, n_atoms=500)
    small = Magnitude(epsilon=4.0, n_atoms=9)
    big = Magnitude(epsilon=4.0, n_atoms=8192)
    assert abs(at_ref.allowance() - 1.0) < 1e-12
    assert 1.40 < small.allowance() < 1.48 and 0.94 < big.allowance() < 0.97
    iv = Interval(t=1.0, k=2, staleness=5.0)
    assert plain.check(iv) is not None and small.check(iv) is None  # 5.0 is noise for 9 atoms, not for 500
    assert small.check(Interval(t=1.0, k=2, staleness=6.0)) is not None


class NoisySmallBackend(BaselineBackend):
    """A few-atom system at k 2: per-interval staleness 3.4 (under epsilon 4.0) with one interval in ``every`` at 5.0,
    above epsilon but under the small-N allowance (4.0 x 1.44 = 5.76); "change" as MDBackend reports it (sqrt(3) x)."""

    def __init__(self, every=7):
        super().__init__(stale_per_step=1.7, probe_stale=1.7)
        self.every, self.attempts = every, 0

    def interval(self, k, mode):
        iv = super().interval(k, mode)
        self.attempts += 1
        iv.staleness = 5.0 if self.attempts % self.every == 0 else 3.4
        iv.extra["change"] = iv.staleness * 3 ** 0.5
        return iv


def test_controller_shrink_uses_the_clause_allowance_one_bound():
    # the DFT ethanol trace (2026-11-15): the clause's allowance let 5.0 pass, but the controller halved k at epsilon,
    # dropped to k 1, and never-lose read k 1 as "the cheap path does not pay": baseline, silently
    allowance = Magnitude(epsilon=4.0, n_atoms=9).allowance()
    assert 1.4 < allowance < 1.5
    for interval_allowance, expect_baseline in ((1.0, True), (allowance, False)):
        b = NoisySmallBackend()
        ctrl = IntervalController(k=2, k_max=2, epsilon=4.0, interval_allowance=interval_allowance)
        ex = Executor(b, [Magnitude(epsilon=4.0, n_atoms=9)], controller=ctrl, never_lose=True, probe_every=50)
        res = ex.run(2000)
        assert not any(e["action"] == "rollback" for e in ex.events)  # the clause passes every interval
        cost = [e for e in ex.events if e["clause"] == "cost-model"]
        if expect_baseline:
            assert cost and res["baseline_steps"] > 500  # the old split bound: k 1, baseline stretches, now logged
        else:
            assert not cost and res["baseline_steps"] == 0


class NoisyRungBackend(BaselineBackend):
    """The DFT ethanol third run's statistics: per-interval staleness ~ N(mean_k, 0.35 mean_k), mean 3.4 at k 2 (under
    epsilon 4.03) and 5.2 at k 3 (over epsilon, under the 9-atom allowance 5.8)."""

    def __init__(self, seed=0):
        import numpy as np

        super().__init__(stale_per_step=1.7, probe_stale=1.7)
        self.rng = np.random.default_rng(seed)

    def interval(self, k, mode):
        iv = super().interval(k, mode)
        mean = {1: 1.8, 2: 3.4, 3: 5.2}.get(k, 7.0)
        iv.staleness = float(max(0.1, self.rng.normal(mean, 0.35 * mean)))
        iv.extra["change"] = iv.staleness * 3 ** 0.5
        return iv


def test_small_n_controller_judges_the_mean_and_grows_only_with_confidence():
    allowance = Magnitude(epsilon=4.03, n_atoms=9).allowance()
    b = NoisyRungBackend()
    ctrl = IntervalController(k=2, k_max=4, epsilon=4.03, interval_allowance=allowance, fine_fs=4)
    ex = Executor(b, [Magnitude(epsilon=4.03, n_atoms=9)], controller=ctrl, never_lose=True, probe_every=50)
    res = ex.run(8000)
    ks = [row[1] for row in ex.log if row[1] > 1]
    rollbacks = sum(e["action"] == "rollback" for e in ex.events)
    assert ks.count(3) <= 0.05 * len(ks)  # the k 3 rung, whose mean passes epsilon, is not where the run lives
    assert rollbacks <= 0.06 * len(ks)  # only k 2's tail past the allowance
    assert res["baseline_steps"] <= 0.1 * 8000  # no never-lose latch


class HotStretchBackend(NoisyRungBackend):
    """As NoisyRungBackend, with a transient hot stretch in simulated time (300-700 fs: every rung's mean x 1.5), as the
    fourth run had at 0.4-0.6 ps; probes during baseline stretches are clean."""

    def interval(self, k, mode):
        iv = super().interval(k, mode)
        if 300 <= self.step <= 700:
            iv.staleness *= 1.5
            iv.extra["change"] = iv.staleness * 3 ** 0.5
        return iv


def test_small_n_controller_recovers_after_a_hot_stretch():
    allowance = Magnitude(epsilon=4.03, n_atoms=9).allowance()
    b = HotStretchBackend()
    ctrl = IntervalController(k=2, k_max=4, epsilon=4.03, interval_allowance=allowance, fine_fs=4)
    ex = Executor(b, [Magnitude(epsilon=4.03, n_atoms=9)], controller=ctrl, never_lose=True, probe_every=50)
    res = ex.run(8000)
    late = [row[1] for row in ex.log if row[0] > 4000]
    assert late.count(2) >= 0.9 * len(late)  # back at k 2 after the stretch, not locked onto the teacher
    assert res["baseline_steps"] <= 0.2 * 8000


def test_halving_from_three_steps_to_two():
    ctrl = IntervalController(k=3, k_max=4, epsilon=2.0, fine_fs=4)
    ctrl.observe(3.0 * 3 ** 0.5, 3)  # h = 3.0 > epsilon
    assert ctrl.k == 2
    ctrl.observe(3.0 * 3 ** 0.5, 2)
    assert ctrl.k == 1  # from 2, halving still reaches 1 (the baseline), as before
