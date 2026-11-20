"""The student factory builds either family and restores saved students exactly."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from frost.neighbours import edges_for
from frost.student import energy_and_forces
from frost.students import load_student, make_student


def frame():
    rng = np.random.default_rng(0)
    cell = np.diag([12.0, 12.3, 12.6])
    grid = np.stack(np.meshgrid(*(np.arange(4),) * 3, indexing="ij"), -1).reshape(-1, 3)
    x = grid * np.array([3.0, 3.075, 3.15]) + rng.normal(0.0, 0.2, grid.shape)
    return x, cell, np.array([14, 8] * 32)


def test_a_saved_frost_student_reloads_to_identical_forces(tmp_path):
    x, cell, numbers = frame()
    torch.manual_seed(0)
    student = make_student("frost", [8, 14], 6.0, cutoff=4.0, channels=8, lmax=3)
    with torch.no_grad():
        for p in student.parameters():
            p.add_(0.1 * torch.randn_like(p))
    path = tmp_path / "student.pt"
    torch.save({"state_dict": student.state_dict(), "config": student.config}, path)
    restored = load_student(path)
    types = student.types(torch.as_tensor(numbers))
    xw, i, j, sh, _ = edges_for(x, cell, 4.0)
    _, f_saved = energy_and_forces(student, xw, types, i, j, sh)
    _, f_restored = energy_and_forces(restored, xw, types, i, j, sh)
    assert torch.equal(f_saved, f_restored)


def test_an_unknown_student_kind_is_rejected():
    with pytest.raises(ValueError):
        make_student("nequip", [8, 14], 6.0)
