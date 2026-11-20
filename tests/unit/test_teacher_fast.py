"""The CUDA-graph teacher must reproduce the eager teacher's forces on every atom."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]


@pytest.fixture(scope="module")
def silicon():
    from ase.build import bulk
    from mace.calculators import mace_mp

    from frost.neighbours import VerletList
    from frost.teacher import MaceTeacher

    atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat(2)
    atoms.rattle(0.05, seed=0)
    calc = mace_mp(model="medium", device="cuda", default_dtype="float32", enable_cueq=True)
    eager = MaceTeacher(calc, atoms, "cuda")
    verlet = VerletList(np.array(atoms.cell), eager.r_max, 1.0, "cuda")
    x = torch.as_tensor(atoms.get_positions(), dtype=torch.float64, device="cuda")
    x, _ = verlet.update(x, force=True)
    eager.set_edges(verlet.i, verlet.j, verlet.shifts, verlet.unit_shifts)
    return atoms, calc, eager, verlet, x


def test_replayed_forces_match_eager_on_every_atom(silicon):
    from frost.teacher_fast import GraphedTeacher

    atoms, calc, eager, verlet, x = silicon
    graphed = GraphedTeacher(calc, atoms, "cuda")
    graphed.set_edges(verlet.i, verlet.j, verlet.shifts, verlet.unit_shifts)
    gen = torch.Generator(device="cuda").manual_seed(0)
    for _ in range(3):
        # every atom moves, well inside half the 1 A skin, so the edges stay valid
        xs = x + 0.05 * torch.randn(x.shape, generator=gen, device="cuda", dtype=x.dtype)
        e_ref, f_ref = eager(xs)
        e_graph, f_graph = graphed(xs)
        assert torch.allclose(f_graph, f_ref, atol=1e-4)
        assert abs(float(e_graph) - float(e_ref)) < 1e-3


def test_energy_only_call_survives_capture_and_matches_eager(silicon):
    """Regression: the captured static output once shadowed the energy() method."""
    from frost.teacher_fast import GraphedTeacher

    atoms, calc, eager, verlet, x = silicon
    graphed = GraphedTeacher(calc, atoms, "cuda")
    graphed.set_edges(verlet.i, verlet.j, verlet.shifts, verlet.unit_shifts)
    graphed(x)  # capture first, then the energy-only path must still work
    assert abs(float(graphed.energy(x)) - float(eager.energy(x))) < 1e-3


def test_graphed_energy_tracks_positions_and_matches_the_eager_forward(silicon):
    """The energy graph must reread the static positions, not replay a frozen value."""
    from frost.teacher_fast import GraphedTeacher

    atoms, calc, eager, verlet, x = silicon
    graphed = GraphedTeacher(calc, atoms, "cuda")
    graphed.set_edges(verlet.i, verlet.j, verlet.shifts, verlet.unit_shifts)
    gen = torch.Generator(device="cuda").manual_seed(1)
    seen = []
    for _ in range(3):
        xs = x + 0.05 * torch.randn(x.shape, generator=gen, device="cuda", dtype=x.dtype)
        e_graph = float(graphed.energy(xs))
        assert abs(e_graph - float(graphed.energy_eager(xs))) < 1e-3
        assert abs(e_graph - float(eager.energy(xs))) < 1e-3
        seen.append(e_graph)
    assert graphed.energy_captures == 1  # captured once, replayed after that
    assert len(set(seen)) == 3  # a stale replay would return the same number every time


def test_a_longer_edge_list_recaptures_the_energy_graph(silicon):
    """Reallocation invalidates both graphs; a stale energy graph would read freed buffers."""
    from frost.neighbours import VerletList
    from frost.teacher_fast import GraphedTeacher

    atoms, calc, eager, verlet, x = silicon
    tight = VerletList(np.array(atoms.cell), eager.r_max, 0.0, "cuda")
    tight.update(x, force=True)
    graphed = GraphedTeacher(calc, atoms, "cuda")

    graphed.set_edges(tight.i, tight.j, tight.shifts, tight.unit_shifts)
    e_tight = float(graphed.energy(x))
    graphed.set_edges(verlet.i, verlet.j, verlet.shifts, verlet.unit_shifts)
    e_skin = float(graphed.energy(x))

    assert graphed.energy_captures == 2
    assert abs(e_tight - float(eager.energy(x))) < 1e-3
    assert abs(e_skin - float(eager.energy(x))) < 1e-3


def test_padding_and_recapture_leave_forces_unchanged(silicon):
    from frost.neighbours import VerletList
    from frost.teacher_fast import GraphedTeacher

    atoms, calc, eager, verlet, x = silicon
    tight = VerletList(np.array(atoms.cell), eager.r_max, 0.0, "cuda")
    tight.update(x, force=True)
    graphed = GraphedTeacher(calc, atoms, "cuda")

    # the tight list leaves padded slots; the 1 A skin list overflows them
    graphed.set_edges(tight.i, tight.j, tight.shifts, tight.unit_shifts)
    f_tight = graphed(x)[1].clone()
    graphed.set_edges(verlet.i, verlet.j, verlet.shifts, verlet.unit_shifts)
    f_skin = graphed(x)[1].clone()

    _, f_ref = eager(x)
    assert graphed.captures == 2
    assert torch.allclose(f_tight, f_ref, atol=1e-4)
    assert torch.allclose(f_skin, f_ref, atol=1e-4)
