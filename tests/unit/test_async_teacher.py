"""The lagged-teacher pipeline, on a one-atom harmonic toy (new_directions.md, C2).

The student is a spring of stiffness K and the teacher one of (1 + mu) K, so the
residual is -mu K x. Everything runs on CPU through the real ``Integrator`` and the
real ``AsyncTeacher`` worker thread; only the force fields are toys. Checks:

- the residual kicked at boundary n is F_T(x_{n-1}) - F_S(x_{n-1}), exactly;
- synchronous boundary r-RESPA keeps the amplitude bounded;
- the one-step lag grows it by 1 + mu theta sin(theta) / 2 per outer step, the
  negative friction derived in experiments/e9_direction/async_theory.py;
- Langevin friction above gamma D = mu theta sin(theta) removes the growth, and
  friction below it does not.
"""

from __future__ import annotations

import math

import pytest
import torch
from ase import units

from frost.async_teacher import AsyncTeacher, LaggedResidual
from frost.integrators import Integrator, Langevin, MDState

MU = 0.05
K_S = 1.0  # eV/A^2; with a 1 amu mass, omega = 1 per ASE time unit
THETA = 0.44 * math.pi  # Frost's operating point, 0.22 of the period
K_INNER = 20
DT_FS = THETA / K_INNER / units.fs
OUTER = 200


@pytest.fixture(autouse=True)
def float64_one_thread():
    threads, dtype = torch.get_num_threads(), torch.get_default_dtype()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(dtype)
    torch.set_num_threads(threads)


def student(x):
    return torch.tensor(0.0), -K_S * x, x


def teacher_call(x_cpu):
    return 0.5 * (1 + MU) * K_S * float((x_cpu * x_cpu).sum()), -(1 + MU) * K_S * x_cpu


def teacher_energy(state):
    return state.kinetic_energy() + 0.5 * (1 + MU) * K_S * float((state.x * state.x).sum())


def fresh(friction_per_fs: float = 0.0):
    state = MDState(torch.tensor([[0.1, 0.0, 0.0]]), torch.zeros(1, 3), torch.ones(1, 1))
    thermostat = Langevin(DT_FS, friction_per_fs, 0.0, state.masses) if friction_per_fs else None
    return state, Integrator(state, DT_FS, thermostat)


def growth_per_outer_step(e_start, e_end, steps):
    return (e_end / e_start) ** (0.5 / steps)  # amplitude, not energy


def run_lagged(friction_per_fs: float = 0.0, depth: int = 1):
    state, integ = fresh(friction_per_fs)
    teacher = AsyncTeacher(lambda: teacher_call)
    try:
        _, f_s, _ = student(state.x)
        residual = teacher(state.x)[1] - f_s
        lag = LaggedResidual(teacher, residual, depth=depth)
        integ.kick(residual, 0.5 * K_INNER * integ.dt)
        e0 = teacher_energy(state)
        seen = []
        for n in range(OUTER):
            for _ in range(K_INNER):
                _, f_s = integ.inner_step(student, f_s)
            kick, finished = lag.boundary(state.x, f_s, state.kinetic_energy(), n)
            seen.append((state.x.clone(), f_s.clone(), kick.clone(), finished))
            integ.kick(kick, K_INNER * integ.dt)
        lag.drain()
        return e0, teacher_energy(state), seen
    finally:
        teacher.close()


def test_the_kick_at_boundary_n_uses_the_call_on_x_n_minus_1():
    _, _, seen = run_lagged()
    for n in range(2, 12):
        x_prev, fs_prev, _, _ = seen[n - 1]
        _, _, kick, finished = seen[n]
        expected = teacher_call(x_prev)[1] - fs_prev
        assert torch.allclose(kick, expected, atol=1e-14)
        assert finished["step"] == n - 1


def test_synchronous_respa_stays_bounded():
    state, integ = fresh()
    _, f_s, _ = student(state.x)
    residual = teacher_call(state.x)[1] - f_s
    e0 = teacher_energy(state)
    for _ in range(OUTER):
        _, f_s = integ.respa_outer(student, f_s, residual, K_INNER)
        residual = teacher_call(state.x)[1] - f_s
        integ.kick(residual, 0.5 * K_INNER * integ.dt)
    assert abs(growth_per_outer_step(e0, teacher_energy(state), OUTER) - 1.0) < 1e-3


def test_a_one_step_lag_is_negative_friction_of_the_predicted_size():
    e0, e1, _ = run_lagged()
    measured = growth_per_outer_step(e0, e1, OUTER) - 1.0
    predicted = 0.5 * MU * THETA * math.sin(THETA)
    assert measured == pytest.approx(predicted, rel=0.15)


@pytest.mark.parametrize("factor, stable", [(1.5, True), (0.5, False)])
def test_langevin_friction_above_the_bound_removes_the_growth(factor, stable):
    gamma_d = factor * MU * THETA * math.sin(THETA)  # per outer step
    friction_per_fs = gamma_d / (K_INNER * DT_FS)
    e0, e1, _ = run_lagged(friction_per_fs)
    growth = growth_per_outer_step(e0, e1, OUTER)
    assert (growth < 1.0) == stable


class ToySpec:
    """Picklable teacher recipe for the process backend (the child imports this module)."""

    def __init__(self, fail: bool = False):
        self.fail = fail

    def __call__(self):
        def call(x_cpu):
            if self.fail and float(x_cpu.abs().sum()) > 1.0:
                raise ValueError("toy teacher refuses large displacements")
            return teacher_call(x_cpu)

        return call


def test_the_process_teacher_answers_in_order_and_exactly():
    from frost.async_teacher import ProcessTeacher

    teacher = ProcessTeacher(ToySpec())
    try:
        xs = [torch.tensor([[0.01 * n, 0.0, 0.0]]) for n in range(5)]
        pending = [teacher.submit(x) for x in xs]
        for x, p in zip(xs, pending, strict=True):
            energy, forces = p.result()
            assert torch.equal(forces, teacher_call(x)[1])
            assert energy == pytest.approx(teacher_call(x)[0])
    finally:
        teacher.close()


def test_a_failure_in_the_child_is_raised_in_the_parent():
    from frost.async_teacher import ProcessTeacher

    teacher = ProcessTeacher(ToySpec(fail=True))
    try:
        assert teacher(torch.tensor([[0.1, 0.0, 0.0]]))[1].shape == (1, 3)
        with pytest.raises(RuntimeError, match="refuses large displacements"):
            teacher(torch.tensor([[5.0, 0.0, 0.0]]))
    finally:
        teacher.close()


class DyingSpec:
    """A child that exits during start-up, as a spawn import error or a crash would."""

    def __call__(self):
        import os

        os._exit(3)


def test_a_dead_child_raises_instead_of_hanging():
    from frost.async_teacher import ProcessTeacher

    with pytest.raises(RuntimeError, match="exited with code 3"):
        ProcessTeacher(DyingSpec())


def test_depth_two_kicks_with_the_call_two_boundaries_back():
    _, _, seen = run_lagged(depth=2)
    for n in range(3, 12):
        x_back, fs_back, _, _ = seen[n - 2]
        _, _, kick, _ = seen[n]
        assert torch.allclose(kick, teacher_call(x_back)[1] - fs_back, atol=1e-14)


def test_a_deeper_lag_grows_the_fastest_mode_less():
    """Growth 1 + mu theta sin(L theta) / 2: at theta = 0.44 pi, depth 2 is 0.37x depth 1."""
    e0, e1, _ = run_lagged(depth=2)
    measured = growth_per_outer_step(e0, e1, OUTER) - 1.0
    predicted = 0.5 * MU * THETA * math.sin(2 * THETA)
    assert measured == pytest.approx(predicted, rel=0.2)
    e0, e1, _ = run_lagged(depth=1)
    assert measured < growth_per_outer_step(e0, e1, OUTER) - 1.0
