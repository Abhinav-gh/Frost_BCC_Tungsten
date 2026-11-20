"""The compiled, CUDA-graph student call must reproduce eager forces on every atom."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from frost.neighbours import edges_for
from frost.student import energy_and_forces
from frost.students import make_student

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]


def test_compiled_graph_matches_eager_forces():
    from frost.student_fast import CompiledGraphedStudent

    rng = np.random.default_rng(0)
    cell = np.diag([12.0, 12.3, 12.6])
    grid = np.stack(np.meshgrid(*(np.arange(4),) * 3, indexing="ij"), -1).reshape(-1, 3)
    x = grid * np.array([3.0, 3.075, 3.15]) + rng.normal(0.0, 0.2, grid.shape)
    numbers = np.array([14, 8] * 32)

    torch.manual_seed(0)
    student = make_student("frost", [8, 14], 6.0, cutoff=4.0, channels=8, lmax=3).cuda()
    with torch.no_grad():
        for p in student.parameters():
            p.add_(0.1 * torch.randn_like(p))
    types = student.types(torch.as_tensor(numbers, device="cuda"))
    xw, i, j, sh, _ = edges_for(x, cell, 4.0, device="cuda")

    runner = CompiledGraphedStudent(student, types, xw.shape[0], capacity=i.numel() + 32)
    runner.set_edges(i, j, sh)
    _, f_compiled = runner(xw)
    _, f_eager = energy_and_forces(student, xw, types, i, j, sh)
    assert torch.allclose(f_compiled, f_eager, atol=1e-4)
