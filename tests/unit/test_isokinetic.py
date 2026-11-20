"""SIN(R) preserves its constraint exactly and samples the right configurations at resonance."""

from __future__ import annotations

import math

import pytest
import torch

from frost.isokinetic import SINR

# the paper's model problem (Leimkuhler, Margul & Tuckerman 2013, section V.A), in
# reduced units: unit mass, beta = 1, U = omega^2 q^2 / 2 + g q^4 / 4
OMEGA, G = 3.0, 0.1


@pytest.fixture(autouse=True)
def one_thread():
    """Thousands of tiny element-wise ops: a thread pool costs more than it saves."""
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def fast(x: torch.Tensor) -> torch.Tensor:
    return -OMEGA * OMEGA * x


def slow(x: torch.Tensor) -> torch.Tensor:
    return -G * x ** 3


def make(n_oscillators: int, n_inner: int, outer: float, L: int = 1, seed: int = 0) -> SINR:
    masses = torch.ones(n_oscillators, 1, dtype=torch.float64)
    sampler = SINR(masses=masses, kT=1.0, dt=outer / n_inner, n_inner=n_inner, L=L,
                   Q1=1.0, Q2=1.0, gamma=1.0, seed=seed)
    gen = torch.Generator().manual_seed(100 + seed)
    sampler.start(torch.randn(n_oscillators, 3, generator=gen, dtype=torch.float64) * 0.3)
    return sampler


def run(sampler: SINR, outer_steps: int, burn: int) -> torch.Tensor:
    """Positions after burn-in, one row per outer step and one column per chain."""
    f = fast(sampler.x)
    s = slow(sampler.x)
    kept = []
    for step in range(outer_steps):
        f, s = sampler.outer_step(fast, f, s, lambda x, _f: slow(x))
        if step >= burn:
            kept.append(sampler.x.flatten().clone())
    return torch.stack(kept)


def l1_to_exact(samples: torch.Tensor, bins: int = 60) -> float:
    edges = torch.linspace(-1.6, 1.6, bins + 1, dtype=torch.float64)
    centres = 0.5 * (edges[1:] + edges[:-1])
    counts = torch.histc(samples.flatten().clamp(-1.6, 1.6), bins=bins, min=-1.6, max=1.6)
    measured = counts / counts.sum()
    exact = torch.exp(-(0.5 * OMEGA ** 2 * centres ** 2 + 0.25 * G * centres ** 4))
    exact = exact / exact.sum()
    return float((measured - exact).abs().sum())


@pytest.mark.parametrize("L", [1, 4])
def test_every_operator_preserves_the_isokinetic_constraint(L):
    """m v^2 + L/(L+1) sum Q1 v1^2 = L kT must hold to rounding after every piece."""
    sampler = make(500, n_inner=10, outer=math.pi / OMEGA, L=L)
    assert sampler.constraint_residual() < 1e-12
    gen = torch.Generator().manual_seed(7)
    for _ in range(50):
        sampler._kick(torch.randn(sampler.v.shape, generator=gen, dtype=torch.float64) * 5.0, 0.03)
        assert sampler.constraint_residual() < 1e-10
        sampler._nose(0.05)
        assert sampler.constraint_residual() < 1e-10
        sampler._ou(0.05)  # touches only v2, which the constraint does not contain
    f, s = fast(sampler.x), slow(sampler.x)
    for _ in range(20):
        f, s = sampler.outer_step(fast, f, s, lambda x, _f: slow(x))
    assert sampler.constraint_residual() < 1e-9


def test_v1_never_changes_sign():
    """Paper Eq. 9: v1 lives in one half-space for all time."""
    sampler = make(300, n_inner=10, outer=math.pi / OMEGA)
    f, s = fast(sampler.x), slow(sampler.x)
    for _ in range(100):
        f, s = sampler.outer_step(fast, f, s, lambda x, _f: slow(x))
    assert bool((sampler.v1 > 0).all())


def exact_q2() -> float:
    q = torch.linspace(-3.0, 3.0, 200001, dtype=torch.float64)
    weight = torch.exp(-(0.5 * OMEGA ** 2 * q * q + 0.25 * G * q ** 4))
    return float((q * q * weight).sum() / weight.sum())


def q2_bias(sampler: SINR, fast_fn, slow_fn, total_time: float, burn_time: float) -> tuple[float, float]:
    """Relative bias of <q^2> and its standard error over independent chains."""
    outer = sampler.n_inner * sampler.dt
    f, s = fast_fn(sampler.x), slow_fn(sampler.x)
    total = torch.zeros_like(sampler.x)
    kept = 0
    for step in range(int(total_time / outer)):
        f, s = sampler.outer_step(fast_fn, f, s, lambda x, _f: slow_fn(x))
        if step >= int(burn_time / outer):
            total += sampler.x * sampler.x
            kept += 1
    per_chain = (total / kept).flatten()
    exact = exact_q2()
    return float(per_chain.mean() / exact - 1.0), float(per_chain.std() / per_chain.numel() ** 0.5 / exact)


def test_samples_the_exact_distribution_at_the_resonant_outer_step():
    """At Delta t = pi/omega an impulse scheme with a weak thermostat resonates; SIN(R) must not."""
    sampler = make(600, n_inner=50, outer=math.pi / OMEGA)
    samples = run(sampler, outer_steps=160, burn=30)
    # the paper reports L1 errors of order 1e-3 to 1e-2 for this problem
    assert l1_to_exact(samples) < 0.03
    per_chain = (samples * samples).mean(dim=0)
    bias = float(per_chain.mean() / exact_q2() - 1.0)
    se = float(per_chain.std() / per_chain.numel() ** 0.5 / exact_q2())
    assert abs(bias) < 4.0 * se


def test_xi_samples_exactly_far_past_resonance():
    """Four half-periods per outer step, the regime the paper's 99 fs water runs used.

    With a slow perturbation XI stays within its error bar (+0.6 +/- 0.4% in <q^2>
    in the sweep that chose it); XO, which integrates the Nose part over half an
    outer step at once, is 24% off here and overflows at twice the step.
    """
    outer = 4.0 * math.pi / OMEGA
    sampler = make(600, n_inner=400, outer=outer, seed=2)
    assert sampler.scheme == "XI"
    bias, se = q2_bias(sampler, fast, slow, 80.0, 10.0)
    assert abs(bias) < 4.0 * se


def test_a_fast_residual_is_not_corrected_at_half_the_period():
    """The limitation that matters for Frost, recorded so nobody rediscovers it on a GPU.

    Frost's slow force is a residual F_T - F_S, and a student with the wrong
    stiffness leaves a residual that oscillates at the fast frequency itself.
    Resonance-free is not bias-free: with a student 5% too stiff (whose own bias in
    <q^2> is -4.8%), SIN(R) at an outer step of half the period is off by -5.4%,
    no better than not running the teacher. A slow residual, as in the paper's water
    runs, is sampled correctly at every outer step tested.
    """
    eps = 0.05

    def stiff(x):
        return -(1.0 + eps) * OMEGA * OMEGA * x

    def residual(x):
        return eps * OMEGA * OMEGA * x - G * x ** 3

    sampler = make(600, n_inner=100, outer=math.pi / OMEGA, seed=3)
    bias, se = q2_bias(sampler, stiff, residual, 80.0, 10.0)
    student_alone = 1.0 / (1.0 + eps) - 1.0
    assert abs(bias) > 5.0 * se
    assert abs(bias) > 0.5 * abs(student_alone)
