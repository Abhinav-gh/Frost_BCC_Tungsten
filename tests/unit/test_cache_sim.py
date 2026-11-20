"""The cache replay must behave correctly before it is trusted on real data.

These run on a synthetic trajectory, so they need no model and no GPU. The
point is to pin the properties the E1 measurement depends on: that phi is
monotone in the tolerance, that the limits are right, that a fixed stride is
recovered exactly, and that CV(tau) is zero for a stride and positive when
edges want different refresh intervals.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from frost.cache_sim import fixed_stride, replay
from frost.predicate import SensitivityTable
from frost.trajectory import TrajectoryReader, TrajectoryWriter


def _synthetic(
    path,
    *,
    n_steps=120,
    n_atoms=32,
    amplitude=0.05,
    cutoff=4.0,
    hot_fraction=0.0,
    seed=0,
):
    """A small cubic crystal whose atoms oscillate.

    `hot_fraction` gives a subset of atoms a much larger amplitude, which is
    the spatial inhomogeneity Frost exists to exploit.
    """
    from ase.build import bulk

    rng = np.random.default_rng(seed)
    at = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))[:n_atoms]
    at.set_cell(bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2)).cell)
    at.pbc = True
    base = at.get_positions()

    amp = np.full(len(at), amplitude)
    if hot_fraction:
        hot = rng.choice(len(at), max(1, int(hot_fraction * len(at))), replace=False)
        amp[hot] = amplitude * 20
    phase = rng.uniform(0, 2 * np.pi, base.shape)
    omega = 0.15

    w = TrajectoryWriter(
        path,
        n_atoms=len(at),
        cutoff=cutoff,
        skin=1.0,
        attrs={
            "cell": np.array(at.cell),
            "numbers": at.get_atomic_numbers(),
            "pbc": np.array(at.pbc),
            "temperature_K": 300.0,
            "timestep_fs": 1.0,
            "cutoff": cutoff,
            "skin": 1.0,
            "seed": seed,
        },
    )
    with w:
        for s in range(n_steps):
            pos = base + amp[:, None] * np.sin(omega * s + phase)
            at.set_positions(pos)
            w.write(s, pos, at, energy=0.0, temperature=300.0)
    return path


@pytest.fixture(scope="module")
def traj(tmp_path_factory):
    p = tmp_path_factory.mktemp("e1") / "synthetic.h5"
    return _synthetic(p)


@pytest.fixture(scope="module")
def hot_traj(tmp_path_factory):
    p = tmp_path_factory.mktemp("e1hot") / "synthetic_hot.h5"
    return _synthetic(p, hot_fraction=0.1, seed=1)


def _table(value, cutoff=4.0):
    return SensitivityTable.constant([14], cutoff=cutoff, value=value)


def test_zero_tolerance_makes_everything_dirty(traj):
    """eps = 0 is the differential-test configuration: every edge, every step."""
    with TrajectoryReader(traj) as r:
        res = replay(r, _table(1.0), eps_meV_per_A=0.0, order=1)
    assert res.phi == pytest.approx(1.0)


def test_huge_tolerance_makes_nothing_dirty(traj):
    with TrajectoryReader(traj) as r:
        res = replay(r, _table(1e-12), eps_meV_per_A=1e6, order=1)
    # only the cold-cache first step is dirty, and phi excludes it
    assert res.phi == pytest.approx(0.0, abs=1e-9)


def test_phi_is_monotone_in_tolerance(traj):
    phis = []
    with TrajectoryReader(traj) as r:
        for eps in (0.1, 1.0, 10.0, 100.0):
            phis.append(replay(r, _table(1.0), eps_meV_per_A=eps, order=1).phi)
    assert all(a >= b - 1e-12 for a, b in pairwise(phis)), phis
    assert phis[0] > phis[-1], "tolerance must actually change the dirty fraction"


def test_fixed_stride_recovers_one_over_k(traj):
    for k in (2, 5, 10):
        with TrajectoryReader(traj) as r:
            res = fixed_stride(r, k=k)
        assert res.phi == pytest.approx(1.0 / k, rel=0.25)
        assert res.cv_tau == 0.0, "a stride has zero refresh-interval spread"


def test_cv_tau_is_positive_when_edges_differ(hot_traj):
    """CV(tau) is the pre-registered statistic for when adaptivity pays.

    With a hot subset, edges want different refresh intervals, so the
    coefficient of variation must be strictly positive.
    """
    with TrajectoryReader(hot_traj) as r:
        res = replay(r, _table(1.0), eps_meV_per_A=5.0, order=1)
    assert res.refresh_interval.size > 0
    assert res.cv_tau > 0.0


def test_age_trigger_bounds_staleness(traj):
    """An edge oscillating below threshold must still refresh eventually."""
    with TrajectoryReader(traj) as r:
        loose = replay(r, _table(1e-12), eps_meV_per_A=1e6, order=1)
        aged = replay(r, _table(1e-12), eps_meV_per_A=1e6, order=1, tau_max=10)
    assert aged.phi > loose.phi
    assert aged.refresh_interval.max() <= 10


def test_replay_is_deterministic(traj):
    with TrajectoryReader(traj) as r:
        a = replay(r, _table(1.0), eps_meV_per_A=2.0, order=1)
        b = replay(r, _table(1.0), eps_meV_per_A=2.0, order=1)
    np.testing.assert_allclose(a.dirty_fraction, b.dirty_fraction)


def test_summary_is_serialisable(traj):
    import json

    with TrajectoryReader(traj) as r:
        res = replay(r, _table(1.0), eps_meV_per_A=2.0, order=1)
    json.dumps(res.summary())  # must not raise on numpy scalars
