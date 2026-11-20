"""The force-only student keeps DMTS-NC's contract: equivariant forces that sum to zero, no energy."""

from __future__ import annotations

import math

import torch

from frost.student import energy_and_forces
from frost.student_nc import ForceStudent


def cluster(n=14, seed=0, cutoff=4.0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 3, generator=gen, dtype=torch.float64) * 5.0
    i, j = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    keep = i != j
    i, j = i[keep], j[keep]
    return x, i, j, torch.zeros(i.numel(), 3, dtype=torch.float64)


def model(species=(1, 8)):
    torch.manual_seed(0)
    m = ForceStudent(list(species), cutoff=4.0).double()
    with torch.no_grad():  # a visible force scale (the head starts near zero)
        m.edge[-1].weight.mul_(30.0)
        m.edge[-1].bias.fill_(0.2)
    return m


def test_net_force_is_zero_and_forces_are_equivariant():
    x, i, j, sh = cluster()
    m = model()
    types = m.types(torch.tensor([1, 8] * 7))
    f = m(x, types, i, j, sh)
    assert f.abs().max() > 1e-3
    assert torch.allclose(f.sum(0), torch.zeros(3, dtype=f.dtype), atol=1e-10)
    a = 0.7
    rot = torch.tensor([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]],
                       dtype=torch.float64)
    f_rot = m(x @ rot.T + 1.3, types, i, j, sh)  # rotated and translated
    assert torch.allclose(f_rot, f @ rot.T, atol=1e-9)


def test_masked_edges_contribute_nothing_and_energy_is_zero():
    x, i, j, sh = cluster()
    m = model()
    types = m.types(torch.tensor([1, 8] * 7))
    mask = torch.ones(i.numel(), dtype=torch.float64)
    far = (x[j] - x[i]).norm(dim=-1) > 3.0
    mask[far] = 0.0
    f_masked = m(x, types, i, j, sh, edge_mask=mask)
    near = ~far
    f_dropped = m(x, types, i[near], j[near], sh[near])
    assert not torch.allclose(f_masked, m(x, types, i, j, sh))  # the mask did something
    # a masked edge changes neither the force head nor the environment moments
    assert torch.allclose(f_masked, f_dropped, atol=1e-10)
    e, f = energy_and_forces(m, x, types, i, j, sh)
    assert float(e.abs().sum()) == 0.0 and torch.allclose(f, m(x, types, i, j, sh))


def test_save_and_load_round_trip(tmp_path):
    from frost.students import load_student

    x, i, j, sh = cluster()
    m = model()
    types = m.types(torch.tensor([1, 8] * 7))
    path = tmp_path / "nc.pt"
    torch.save({"state_dict": m.state_dict(), "config": m.config}, path)
    back = load_student(path).double()
    assert back.returns_forces
    assert torch.allclose(back(x, types, i, j, sh), m(x, types, i, j, sh), atol=1e-12)


def test_the_trainer_fits_forces_without_an_energy():
    from frost.insitu import Frame, InSituTrainer, LabelBuffer

    x, i, j, sh = cluster(n=10)
    teacher = model()
    student = ForceStudent([1, 8], cutoff=4.0).double()
    types = student.types(torch.tensor([1, 8] * 5))
    buffer = LabelBuffer()
    gen = torch.Generator().manual_seed(1)
    for step in range(8):
        xs = x + 0.05 * torch.randn(x.shape, generator=gen, dtype=torch.float64)
        with torch.no_grad():
            f_t = teacher(xs, types, i, j, sh)
        buffer.add(Frame(step, xs, f_t, 0.0, i, j, sh))
    trainer = InSituTrainer(student, types, lr=5e-3, max_gap=2)
    first = trainer.step(buffer, batch_pairs=4, report=True)["force_rmse_meV_per_A"]
    for _ in range(150):
        trainer.step(buffer, batch_pairs=4)
    last = trainer.step(buffer, batch_pairs=4, report=True)["force_rmse_meV_per_A"]
    assert last < 0.7 * first
