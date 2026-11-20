"""MACE students on cuEquivariance or OpenEquivariance kernels are the e3nn student, faster.

Every converted student must reproduce the e3nn student's forces on every atom,
eager and inside a CUDA graph, and must train through a force loss (double
backward through the fused kernels).
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")  # e3nn 0.4.4 on torch >= 2.6
pytest.importorskip("mace")

from frost.neighbours import edges_for
from frost.student import GraphedStudent, energy_and_forces
from frost.student_mace import MaceStudent

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]
BACKENDS = ["cueq", "oeq"]


def _system():
    rng = np.random.default_rng(0)
    cell = np.diag([12.0, 12.3, 12.6])
    grid = np.stack(np.meshgrid(*(np.arange(4),) * 3, indexing="ij"), -1).reshape(-1, 3)
    x = grid * np.array([3.0, 3.075, 3.15]) + rng.normal(0.0, 0.2, grid.shape)
    numbers = np.array([14, 8] * 32)
    torch.manual_seed(0)
    student = MaceStudent([8, 14], cutoff=4.0, channels=8, max_ell=2, layers=2, correlation=3,
                          mean_neighbours=6.0).cuda()
    with torch.no_grad():
        for p in student.parameters():
            p.add_(0.1 * torch.randn_like(p))
    types = student.types(torch.as_tensor(numbers, device="cuda"))
    xw, i, j, sh, _ = edges_for(x, cell, 4.0, device="cuda")
    return student, types, xw, i, j, sh


@pytest.mark.parametrize("backend", BACKENDS)
def test_converted_student_matches_e3nn_on_every_atom(backend):
    student, types, x, i, j, sh = _system()
    e_ref, f_ref = energy_and_forces(student, x, types, i, j, sh)
    fast = student.converted(backend)
    assert fast.config["backend"] == backend
    e_fast, f_fast = energy_and_forces(fast, x, types, i, j, sh)
    assert torch.allclose(f_fast, f_ref, atol=1e-4)
    assert abs(float(e_fast.sum() - e_ref.sum())) < 1e-3


@pytest.mark.parametrize("backend", BACKENDS)
def test_graphed_converted_student_matches_e3nn(backend):
    student, types, x, i, j, sh = _system()
    _, f_ref = energy_and_forces(student, x, types, i, j, sh)
    runner = GraphedStudent(student.converted(backend), types, x.shape[0], capacity=i.numel() + 32)
    runner.set_edges(i, j, sh)
    _, f_graph = runner(x)
    assert torch.allclose(f_graph, f_ref, atol=1e-4)


def test_autocast_student_keeps_the_calling_contract():
    from frost.student_fast import AutocastStudent

    student, types, x, i, j, sh = _system()
    wrapped = AutocastStudent(student.converted("cueq"))
    energy, forces = energy_and_forces(wrapped, x, types, i, j, sh)
    assert energy.dtype == torch.float32 and energy.shape == (1,)
    assert forces.shape == x.shape and bool(torch.isfinite(forces).all())
    assert wrapped.config["autocast"] == "bfloat16"


@pytest.mark.parametrize("backend", BACKENDS)
def test_converted_student_trains_through_a_force_loss(backend):
    student, types, x, i, j, sh = _system()
    fast = student.converted(backend)
    before = [p.detach().clone() for p in fast.parameters()]
    opt = torch.optim.Adam(fast.parameters(), lr=1e-3)
    _, f = energy_and_forces(fast, x, types, i, j, sh, create_graph=True)
    loss = f.pow(2).mean()
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
    assert any(not torch.equal(b, p.detach())
               for b, p in zip(before, fast.parameters(), strict=True))
