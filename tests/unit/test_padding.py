"""Spare edge slots are spread over the atoms, never piled on one (frost.padding)."""

from __future__ import annotations

import torch

from frost.padding import spread_padding


def test_spare_slots_repeat_no_index_more_than_ceil_spare_over_atoms():
    pad = spread_padding(100, 1_433, 500, "cpu")
    assert pad.numel() == 1_333
    counts = torch.bincount(pad, minlength=500)
    assert int(counts.max()) == -(-1_333 // 500)  # ceil
    assert int(pad.min()) >= 0 and int(pad.max()) < 500


def test_a_full_buffer_has_no_spare_slots():
    assert spread_padding(64, 64, 10, "cpu").numel() == 0
    assert spread_padding(80, 64, 10, "cpu").numel() == 0


def test_the_graphed_student_spreads_its_spare_slots_and_keeps_its_forces():
    from frost.student import FrostStudent, GraphedStudent, energy_and_forces

    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA graphs need a GPU")
    torch.manual_seed(0)
    n = 12
    x = (torch.rand(n, 3) * 6.0).cuda()
    i, j = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    keep = i != j
    i, j = i[keep].cuda(), j[keep].cuda()
    sh = torch.zeros(i.numel(), 3).cuda()
    model = FrostStudent(species=[29], cutoff=8.0).float().cuda()
    types = model.types(torch.full((n,), 29).cuda())
    g = GraphedStudent(model, types, n, capacity=i.numel() + 40)
    g.set_edges(i, j, sh)
    assert int(torch.bincount(g.i[i.numel():], minlength=n).max()) <= -(-40 // n)
    _, f_graph = g(x)
    _, f_eager = energy_and_forces(model, x, types, i, j, sh)
    assert torch.allclose(f_graph, f_eager, atol=1e-4)
