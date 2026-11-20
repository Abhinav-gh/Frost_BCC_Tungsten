"""The student potential: conservative, invariant, strictly local, batchable, trainable."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from frost.insitu import Frame, InSituTrainer, LabelBuffer
from frost.neighbours import edges_for
from frost.student import FrostStudent, energy_and_forces

CUTOFF = 4.0


def lattice(seed: int = 0, jitter: float = 0.2):
    rng = np.random.default_rng(seed)
    cell = np.diag([12.0, 12.3, 12.6])
    grid = np.stack(np.meshgrid(*(np.arange(4),) * 3, indexing="ij"), -1).reshape(-1, 3)
    x = grid * np.array([3.0, 3.075, 3.15]) + rng.normal(0.0, jitter, grid.shape)
    return x, cell, np.array([14, 8] * 32)


def random_model(seed: int = 0, lmax: int = 2) -> FrostStudent:
    """Randomised weights, normalised as in production so invariants stay O(1)."""
    torch.manual_seed(seed)
    model = FrostStudent([8, 14], cutoff=CUTOFF, lmax=lmax).double()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.3 * torch.randn_like(p))
    model.set_neighbour_normalisation(6.0)
    return model


def setup(seed: int = 0, cutoff: float = CUTOFF, lmax: int = 2):
    x, cell, numbers = lattice(seed)
    model = random_model(seed, lmax)
    types = model.types(torch.as_tensor(numbers))
    xw, i, j, sh, _ = edges_for(x, cell, cutoff, dtype=torch.float64)
    return model, types, xw, i, j, sh


def run(model, types, x, i, j, sh, **kw):
    e, f = energy_and_forces(model, x, types, i, j, sh, **kw)
    return e.detach(), f.detach()


@pytest.mark.parametrize("lmax", [2, 3])
def test_forces_are_the_negative_energy_gradient_along_an_all_atom_direction(lmax):
    model, types, x, i, j, sh = setup(lmax=lmax)
    _, f = run(model, types, x, i, j, sh)
    direction = torch.randn(x.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    h = 1e-5
    e_plus, _ = run(model, types, x + h * direction, i, j, sh)
    e_minus, _ = run(model, types, x - h * direction, i, j, sh)
    numeric = -(e_plus - e_minus).item() / (2 * h)
    assert (f * direction).sum().item() == pytest.approx(numeric, rel=1e-5)


@pytest.mark.parametrize("lmax", [2, 3])
def test_energy_is_invariant_and_forces_rotate_under_rotation(lmax):
    model, types, x, i, j, sh = setup(lmax=lmax)
    e, f = run(model, types, x, i, j, sh)
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64,
                                       generator=torch.Generator().manual_seed(2)))
    e_rot, f_rot = run(model, types, x @ q.T, i, j, sh @ q.T)
    assert e_rot.item() == pytest.approx(e.item(), rel=1e-9)
    assert torch.allclose(f_rot, f @ q.T, rtol=1e-7, atol=1e-9)


def test_translation_changes_nothing():
    model, types, x, i, j, sh = setup()
    e, f = run(model, types, x, i, j, sh)
    e_t, f_t = run(model, types, x + torch.tensor([0.3, -1.2, 2.5], dtype=torch.float64), i, j, sh)
    assert e_t.item() == pytest.approx(e.item(), rel=1e-12)
    assert torch.allclose(f_t, f, atol=1e-10)


@pytest.mark.parametrize("lmax", [2, 3])
def test_permuting_atoms_permutes_forces(lmax):
    model, types, x, i, j, sh = setup(lmax=lmax)
    e, f = run(model, types, x, i, j, sh)
    perm = torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(3))
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(x.shape[0])
    e_p, f_p = run(model, types[perm], x[perm], inverse[i], inverse[j], sh)
    assert e_p.item() == pytest.approx(e.item(), rel=1e-12)
    assert torch.allclose(f_p, f[perm], atol=1e-10)


def test_edges_beyond_the_cutoff_contribute_nothing():
    model, types, x, i, j, sh = setup()
    _, i_skin, j_skin, sh_skin, _ = edges_for(lattice()[0], lattice()[1], CUTOFF + 1.5,
                                              dtype=torch.float64)
    assert i_skin.numel() > i.numel()
    e, f = run(model, types, x, i, j, sh)
    e_skin, f_skin = run(model, types, x, i_skin, j_skin, sh_skin)
    assert e_skin.item() == pytest.approx(e.item(), rel=1e-12)
    assert torch.allclose(f_skin, f, atol=1e-10)


def test_padded_edges_are_inert():
    model, types, x, i, j, sh = setup(lmax=3)
    pad = 40
    zeros = torch.zeros(pad, dtype=torch.long)
    mask = torch.cat((torch.ones(i.numel()), torch.zeros(pad))).double()
    e, f = run(model, types, x, i, j, sh)
    e_pad, f_pad = run(model, types, x, torch.cat((i, zeros)), torch.cat((j, zeros)),
                       torch.cat((sh, torch.zeros(pad, 3, dtype=torch.float64))), edge_mask=mask)
    assert e_pad.item() == pytest.approx(e.item(), rel=1e-12)
    assert torch.allclose(f_pad, f, atol=1e-10)


def test_a_batch_of_frames_matches_frames_one_at_a_time():
    model, types, x, i, j, sh = setup(seed=0)
    _, _, x2, i2, j2, sh2 = setup(seed=4)
    n = x.shape[0]
    e1, f1 = run(model, types, x, i, j, sh)
    e2, f2 = run(model, types, x2, i2, j2, sh2)
    batch = torch.arange(2).repeat_interleave(n)
    e_b, f_b = run(model, types.repeat(2), torch.cat((x, x2)), torch.cat((i, i2 + n)),
                   torch.cat((j, j2 + n)), torch.cat((sh, sh2)), batch=batch, n_graphs=2)
    assert torch.allclose(e_b, torch.cat((e1, e2)), rtol=1e-12)
    assert torch.allclose(f_b, torch.cat((f1, f2)), atol=1e-10)


def test_training_on_a_stream_of_teacher_calls_reduces_the_loss():
    teacher = random_model(seed=5)
    student = FrostStudent([8, 14], cutoff=CUTOFF).double()
    x, cell, numbers = lattice(seed=6)
    types = student.types(torch.as_tensor(numbers))
    rng = np.random.default_rng(7)
    buffer = LabelBuffer()
    for step in range(8):
        x = x + rng.normal(0.0, 0.02, x.shape)
        xw, i, j, sh, _ = edges_for(x, cell, CUTOFF, dtype=torch.float64)
        e_t, f_t = run(teacher, teacher.types(torch.as_tensor(numbers)), xw, i, j, sh)
        buffer.add(Frame(step, xw, f_t, float(e_t.sum()), i, j, sh))
    trainer = InSituTrainer(student, types, lr=1e-2, max_gap=1)
    first = trainer.step(buffer, batch_pairs=4, report=True)["loss"]
    for _ in range(60):
        trainer.step(buffer, batch_pairs=4)
    last = trainer.step(buffer, batch_pairs=4, report=True)["loss"]
    assert last < 0.7 * first


def test_ring_pairs_teach_the_force_change_across_beads():
    """Ring pairs (bead dumps, 2026-11-17): frames `stride` apart are the same time on two beads; with ring_batch the
    change terms include them, and training on them shrinks the student's error on the teacher's bead-to-bead force
    difference."""
    teacher = random_model(seed=5)
    student = FrostStudent([8, 14], cutoff=CUTOFF).double()
    x0, cell, numbers = lattice(seed=6)
    types = student.types(torch.as_tensor(numbers))
    t_types = teacher.types(torch.as_tensor(numbers))
    rng = np.random.default_rng(7)
    stride, buffer = 100, LabelBuffer()
    beads = [x0 + rng.normal(0.0, 0.05, x0.shape) for _ in range(2)]
    for b, xb in enumerate(beads):  # two bead blocks, four times each
        for t in range(4):
            x = xb + 0.01 * t
            xw, i, j, sh, _ = edges_for(x, cell, CUTOFF, dtype=torch.float64)
            e_t, f_t = run(teacher, t_types, xw, i, j, sh)
            buffer.add(Frame(b * stride + t, xw, f_t, float(e_t.sum()), i, j, sh))
    assert sorted(buffer.ring_pairs(stride)) == [(0, 4), (1, 5), (2, 6), (3, 7)]

    def ring_error():
        errs = []
        for a, b in buffer.ring_pairs(stride):
            fa, fb = buffer.frames[a], buffer.frames[b]
            sa = run(student, types, fa.x, fa.edge_i, fa.edge_j, fa.shifts)[1]
            sb = run(student, types, fb.x, fb.edge_i, fb.edge_j, fb.shifts)[1]
            errs.append(float(((sb - sa) - (fb.forces - fa.forces)).pow(2).mean()))
        return sum(errs) / len(errs)

    trainer = InSituTrainer(student, types, lr=1e-2, max_gap=1, w_force=0.0, w_denergy=0.0, ring_stride=stride,
                            ring_batch=4)
    before = ring_error()
    for _ in range(60):
        trainer.step(buffer, batch_pairs=1)
    assert ring_error() < 0.7 * before


def test_species_balance_weights_every_element_equally():
    """Weights average to one over the atoms and give each element the same total (2026-11-17)."""
    student = FrostStudent([8, 14], cutoff=CUTOFF).double()
    types = student.types(torch.as_tensor([14] * 30 + [8] * 2))
    w = InSituTrainer(student, types, species_balance=True).atom_w
    assert abs(float(w.mean()) - 1.0) < 1e-12
    assert abs(float(w[:30].sum()) - float(w[30:].sum())) < 1e-9
    assert InSituTrainer(student, types).atom_w is None


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_the_cuda_graph_matches_eager_evaluation():
    from frost.student import GraphedStudent

    model, types, x, i, j, sh = setup(lmax=3)
    model = model.float().cuda()
    types, x, i, j, sh = types.cuda(), x.float().cuda(), i.cuda(), j.cuda(), sh.float().cuda()
    graphed = GraphedStudent(model, types, x.shape[0], capacity=i.numel() + 32)
    graphed.set_edges(i, j, sh)
    _, f_graph = graphed(x)
    _, f_eager = energy_and_forces(model, x, types, i, j, sh)
    assert torch.allclose(f_graph, f_eager, atol=1e-4)
