"""Lists from unwrapped positions give the wrapped list's vectors, on either builder."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from frost.neighbours import VerletList, edges_for


def _positions(seed: int = 0, n: int = 96):
    """A triclinic cell with a third of the atoms pushed one or two cells away."""
    rng = np.random.default_rng(seed)
    cell = np.array([[12.0, 0.0, 0.0], [1.5, 12.3, 0.0], [0.7, -0.4, 12.6]])
    x = rng.random((n, 3)) @ cell
    jumps = rng.integers(-2, 3, size=(n, 3)) * (rng.random((n, 1)) < 0.33)
    return cell, x + jumps @ cell


def _sorted_edges(x, i, j, shifts):
    """(i, j, r_ij) rows on the CPU, in a canonical order."""
    x, i, j, shifts = (t.detach().cpu() for t in (x, i, j, shifts))
    r = (x[j] - x[i] + shifts).double().numpy()
    order = np.lexsort((*np.round(r, 4).T[::-1], j.numpy(), i.numpy()))
    return i.numpy()[order], j.numpy()[order], r[order]


def _assert_same_edges(a, b):
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert np.allclose(a[2], b[2], atol=1e-8)


def test_unwrapped_list_reproduces_the_wrapped_vectors():
    cell, x = _positions()
    wrapped = edges_for(x, cell, 4.5, dtype=torch.float64)
    xu, i, j, shifts, unit = edges_for(x, cell, 4.5, dtype=torch.float64, wrap_positions=False)
    assert torch.equal(xu, torch.as_tensor(x))
    assert torch.allclose(shifts, unit.double() @ torch.as_tensor(cell))
    _assert_same_edges(_sorted_edges(*wrapped[:4]), _sorted_edges(xu, i, j, shifts))


def test_verlet_list_without_wrapping_returns_positions_unchanged():
    cell, x = _positions(1)
    xt = torch.as_tensor(x)
    verlet = VerletList(cell, 4.0, 0.5, "cpu", wrap_positions=False)
    x_out, rebuilt = verlet.update(xt)
    assert rebuilt and x_out is xt
    reference = edges_for(x, cell, 4.5, dtype=torch.float64)
    _assert_same_edges(_sorted_edges(*reference[:4]),
                       _sorted_edges(xt, verlet.i, verlet.j, verlet.shifts))


def test_unknown_builder_is_refused():
    with pytest.raises(ValueError):
        VerletList(np.eye(3) * 10.0, 4.0, builder="kdtree")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("wrap_positions", [True, False])
def test_gpu_builder_matches_vesin_on_every_edge(wrap_positions, dtype):
    from frost.neighbours import edges_on_gpu

    cell, x = _positions(2, n=400)
    ref = edges_for(x, cell, 5.0, dtype=torch.float64, wrap_positions=wrap_positions)
    got = edges_on_gpu(torch.as_tensor(x, dtype=dtype, device="cuda"), cell, 5.0,
                       wrap_positions=wrap_positions)
    a = _sorted_edges(*ref[:4])
    b = _sorted_edges(got[0].double(), got[1], got[2], got[3].double())
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert np.allclose(a[2], b[2], atol=1e-8 if dtype == torch.float64 else 1e-4)


def _edge_set(lst):
    return set(zip(lst.i.tolist(), lst.j.tolist(), map(tuple, lst.unit_shifts.long().tolist())))


def _random_cell(n=60, seed=0):
    rng = np.random.default_rng(seed)
    cell = np.diag([9.0, 9.5, 10.0])
    return rng.uniform(0, 1, (n, 3)) @ cell, cell


def test_derived_list_has_exactly_the_direct_build_edges():
    x_np, cell = _random_cell()
    x = torch.as_tensor(x_np)
    outer = VerletList(cell, 6.0, 1.0, "cpu", wrap_positions=False)
    inner = VerletList(cell, 3.5, 0.3, "cpu", wrap_positions=False, outer=outer)
    direct = VerletList(cell, 3.5, 0.3, "cpu", wrap_positions=False)
    for step in range(3):  # build, then rebuild after moving past the inner skin
        outer.update(x, force=(step == 0))
        inner.update(x, force=True)
        direct.update(x, force=True)
        assert _edge_set(inner) == _edge_set(direct)
        x = x + 0.2 * torch.randn_like(x)


def test_derived_list_refuses_a_radius_beyond_the_outer_cutoff():
    _, cell = _random_cell()
    outer = VerletList(cell, 4.0, 0.5, "cpu", wrap_positions=False)
    with pytest.raises(ValueError):
        VerletList(cell, 3.8, 0.5, "cpu", wrap_positions=False, outer=outer)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_lagged_list_never_misses_a_pair_within_the_cutoff():
    x_np, cell = _random_cell(n=80, seed=1)
    x = torch.as_tensor(x_np, device="cuda")
    v = 0.01 * torch.randn_like(x)  # ~0.01 A per step: under the 0.05 A margin
    lagged = VerletList(cell, 3.5, 0.3, "cuda", wrap_positions=False, lag_margin=0.05)
    lagged.update(x, force=True)
    for _ in range(200):
        x = x + v
        lagged.update(x)
        fresh = VerletList(cell, 3.5, 0.0, "cuda", wrap_positions=False)
        fresh.update(x, force=True)
        assert _edge_set(fresh) <= _edge_set(lagged)
    assert lagged.rebuilds > 1


@pytest.mark.parametrize("wrap_positions", [True, False])
def test_brute_builder_matches_vesin_on_every_edge(wrap_positions):
    from frost.neighbours import edges_brute

    rng = np.random.default_rng(5)
    cell = np.diag([12.0, 13.0, 14.0])
    x = rng.random((300, 3)) @ cell + rng.integers(-2, 3, size=(300, 3)) * (rng.random((300, 1)) < 0.33) @ cell
    ref = edges_for(x, cell, 5.0, dtype=torch.float64, wrap_positions=wrap_positions)
    got = edges_brute(torch.as_tensor(x), cell, 5.0, wrap_positions=wrap_positions)
    _assert_same_edges(_sorted_edges(*ref[:4]), _sorted_edges(*got[:4]))
    assert torch.equal(got[3], got[4].double() @ torch.as_tensor(cell))


def test_brute_builder_refuses_what_it_cannot_do_and_auto_falls_back():
    from frost.neighbours import brute_ok, edges_brute

    cell, x = _positions()  # triclinic
    assert not brute_ok(cell, 4.0)
    with pytest.raises(ValueError):
        edges_brute(torch.as_tensor(x), cell, 4.0)
    assert not brute_ok(np.diag([9.0, 9.0, 9.0]), 5.0)  # radius over half the box: two images may count
    auto = VerletList(cell, 4.0, 0.5, "cpu", wrap_positions=False, builder="auto")
    auto.update(torch.as_tensor(x))  # falls back to vesin on a triclinic cell
    ref = edges_for(x, cell, 4.5, dtype=torch.float64, wrap_positions=False)
    _assert_same_edges(_sorted_edges(*ref[:4]), _sorted_edges(torch.as_tensor(x), auto.i, auto.j, auto.shifts))


def test_tight_filter_over_copies_keeps_exactly_each_copys_pairs_within_the_cutoff():
    """concat_lists + filter_within (path-integral beads, 2026-11-17): the same edges as tight_edges per copy, offset by
    b x N, and exactly the pairs within r_max at each copy's own positions, from lists built at other positions (every
    atom within half the skin of where its list was built)."""
    from frost.neighbours import concat_lists, filter_within, tight_edges

    rng = np.random.default_rng(3)
    box, n, copies, r_max, skin = 9.0, 48, 3, 3.0, 1.0
    cell = np.eye(3) * box
    base = torch.as_tensor(rng.random((n, 3)) * box)
    lists, q = [], []
    for _ in range(copies):
        lst = VerletList(cell, r_max, skin, "cpu", wrap_positions=False)
        lst.update(base, force=True)
        lists.append(lst)
        step = torch.as_tensor(rng.normal(size=(n, 3)))
        step = step / step.norm(dim=1, keepdim=True) * torch.as_tensor(0.45 * skin * rng.random((n, 1)))
        q.append(base + step)
    q = torch.stack(q)
    x = q.reshape(-1, 3)
    i, j, sh, us = filter_within(*concat_lists(lists, n), x, r_max ** 2)
    per = [tight_edges(lst, q[b], r_max ** 2) for b, lst in enumerate(lists)]
    assert torch.equal(i, torch.cat([e[0] + b * n for b, e in enumerate(per)]))
    assert torch.equal(j, torch.cat([e[1] + b * n for b, e in enumerate(per)]))
    assert torch.equal(sh, torch.cat([e[2] for e in per])) and torch.equal(us, torch.cat([e[3] for e in per]))
    for b in range(copies):
        d = q[b][None, :, :] - q[b][:, None, :]
        d = d - box * torch.round(d / box)  # r_max < box / 2: one image per pair
        within = (d.norm(dim=-1) < r_max) & ~torch.eye(n, dtype=torch.bool)
        mine = (i >= b * n) & (i < (b + 1) * n)
        assert bool(((j >= b * n) & (j < (b + 1) * n))[mine].all())
        assert int(mine.sum()) == int(within.sum()) > 0
        assert float((x[j[mine]] - x[i[mine]] + sh[mine]).norm(dim=1).max()) < r_max
