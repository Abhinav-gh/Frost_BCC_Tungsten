"""Nested HMC samples the teacher's distribution exactly, whatever the student gets wrong."""

from __future__ import annotations

import math

import pytest
import torch
from ase import units

from frost.nested import NestedHMC

T, MASS, K_TEACHER = 300.0, 16.0, 10.0
TARGET = units.kB * T / K_TEACHER


def spring(kappa: float):
    def energy(x):
        return 0.5 * kappa * (x * x).sum()

    def force(x):
        return energy(x), -kappa * x, x

    return energy, force


def chain(n: int, k_student: float, refresh: float, steps: int, seed: int,
          extra_chances: int = 0, jitter: float = 0.0):
    """Mean of x^2 after burn-in, relative to the teacher's variance, and the acceptance."""
    teacher_energy, _ = spring(K_TEACHER)
    _, student_force = spring(k_student)
    masses = torch.full((n, 1), MASS, dtype=torch.float64)
    sampler = NestedHMC(masses, T, dt_fs=2.0, n_draft=8, student_force=student_force,
                        teacher_energy=teacher_energy, refresh=refresh, seed=seed,
                        extra_chances=extra_chances, jitter=jitter)
    # start from the student's distribution, so a chain that did not mix would fail
    g = torch.Generator().manual_seed(100 + seed)
    x = torch.randn(n, 3, generator=g, dtype=torch.float64) * (units.kB * T / k_student) ** 0.5
    v = torch.zeros_like(x)
    e, f = float(teacher_energy(x)), student_force(x)[1]
    total, count = 0.0, 0
    for step in range(steps):
        x, v, e, f, _ = sampler.step(x, v, e, f)
        if step >= steps // 10:
            total += float(x.pow(2).mean())
            count += 1
    return total / count / TARGET, sampler.acceptance


@pytest.mark.parametrize("refresh", [1.0, 0.5])
def test_a_wrong_student_still_samples_the_teacher_variance(refresh):
    """Drafts use a student 30% too stiff; the variance must be the teacher's, not the student's."""
    ratios = [chain(16, 13.0, refresh, 12000, seed)[0] for seed in (1, 2)]
    student_ratio = K_TEACHER / 13.0  # 0.77, where a biased sampler would drift
    assert sum(ratios) / len(ratios) == pytest.approx(1.0, abs=0.02)
    assert all(abs(r - student_ratio) > 0.15 for r in ratios)


def test_acceptance_falls_as_the_proposal_grows():
    """The energy error of a whole-system proposal is extensive (note section 5.3)."""
    small = chain(16, 13.0, 1.0, 1500, 3)[1]
    large = chain(256, 13.0, 1.0, 1500, 3)[1]
    assert small > 0.2
    assert large < 0.05


def test_a_perfect_student_accepts_almost_everything():
    energy, force = spring(K_TEACHER)
    masses = torch.full((50, 1), MASS, dtype=torch.float64)
    sampler = NestedHMC(masses, T, dt_fs=0.5, n_draft=4, student_force=force,
                        teacher_energy=energy, seed=3)
    x = torch.zeros(50, 3, dtype=torch.float64)
    v = torch.zeros_like(x)
    e, f = float(energy(x)), force(x)[1]
    for _ in range(200):
        x, v, e, f, _ = sampler.step(x, v, e, f)
    assert sampler.acceptance > 0.95


def test_extra_chance_probabilities_satisfy_detailed_balance():
    """pi(z_0) P_k(z_0) = pi(z_k) P_k(F z_k) for every k, on random energy sequences."""
    sampler = NestedHMC(torch.ones(1, 1, dtype=torch.float64), T, dt_fs=1.0, n_draft=1,
                        student_force=None, teacher_energy=None)
    kT = sampler.kT
    g = torch.Generator().manual_seed(0)
    for _ in range(200):
        h = (torch.randn(5, generator=g, dtype=torch.float64) * 2.0 * kT).tolist()
        forward = sampler.chance_probabilities(h)
        assert sum(forward) <= 1.0 + 1e-12
        assert all(p >= 0.0 for p in forward)
        for k in range(1, len(h)):
            reverse = sampler.chance_probabilities(h[k::-1])  # the flipped chain from z_k
            lhs = math.exp(-h[0] / kT) * forward[k - 1]
            rhs = math.exp(-h[k] / kT) * reverse[k - 1]
            assert lhs == pytest.approx(rhs, rel=1e-9, abs=1e-300)


def test_no_extra_chances_is_plain_metropolis():
    sampler = NestedHMC(torch.ones(1, 1, dtype=torch.float64), T, dt_fs=1.0, n_draft=1,
                        student_force=None, teacher_energy=None)
    for dh in (-1.0, 0.0, 0.5, 3.0):
        p = sampler.chance_probabilities([0.0, dh * sampler.kT])
        assert p == [pytest.approx(min(1.0, math.exp(-dh)))]


@pytest.mark.parametrize("extra_chances,jitter,refresh", [(2, 0.0, 1.0), (0, 0.5, 1.0), (2, 0.5, 0.5)])
def test_extra_chances_and_random_length_stay_exact(extra_chances, jitter, refresh):
    """A 30%-stiff student with the new kernels must still give the teacher's variance."""
    ratios = [chain(16, 13.0, refresh, 8000, seed, extra_chances, jitter)[0] for seed in (1, 2)]
    student_ratio = K_TEACHER / 13.0
    assert sum(ratios) / len(ratios) == pytest.approx(1.0, abs=0.02)
    assert all(abs(r - student_ratio) > 0.15 for r in ratios)


def test_extra_chances_raise_the_fraction_of_proposals_that_move():
    plain = chain(64, 13.0, 1.0, 1500, 4)[1]
    rescued = chain(64, 13.0, 1.0, 1500, 4, extra_chances=3)[1]
    assert rescued > plain + 0.05
