"""frost.locality: the magnitude clause measured region by region."""

import math

import torch

from frost.locality import region_factor, region_index, staleness_stats


def _cell(n_atoms=8192, side=50.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n_atoms, 3, generator=g, dtype=torch.float64) * side
    return x, torch.full((3,), side, dtype=torch.float64), g


def test_region_index_wraps_and_covers():
    x, box, _ = _cell()
    idx = region_index(x, box, 4)
    assert int(idx.min()) == 0 and int(idx.max()) == 63
    assert torch.equal(idx, region_index(x + box * torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64), box, 4))


def test_homogeneous_staleness_leaves_the_whole_cell_clause_in_charge():
    # spread evenly, no region crosses its allowance before the whole cell crosses epsilon
    x, box, g = _cell()
    for seed in range(100, 120):  # not the positions' seed: the same stream would correlate da with x
        da = torch.randn(len(x), 3, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
        st = staleness_stats(da, x, box, 4)
        assert st["local"] < st["rms"], (seed, st)
        assert math.isclose(st["rms"], float((da ** 2).mean().sqrt()), rel_tol=1e-12)


def test_concentrated_staleness_is_seen_locally():
    # 100 atoms in one corner region 9x staler than the rest: the whole-cell RMS rises by ~40%, the region by ~8x
    x, box, g = _cell()
    da = torch.randn(len(x), 3, generator=g, dtype=torch.float64)
    hot = torch.nonzero(region_index(x, box, 4) == 0).squeeze(1)[:100]
    da[hot] *= 9.0
    st = staleness_stats(da, x, box, 4)
    assert st["rms"] < 1.6
    assert st["local"] > 4.0 * st["rms"]
    assert st["atoms"] >= 100


def test_full_cell_matches_lengths_and_handles_a_skewed_cell():
    x, box, _ = _cell()
    assert torch.equal(region_index(x, box, 4), region_index(x, torch.diag(box), 4))
    cell = torch.tensor([[50.0, 0.0, 0.0], [25.0, 43.3, 0.0], [0.0, 0.0, 40.0]], dtype=torch.float64)
    frac = torch.rand(4096, 3, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    idx = region_index(frac @ cell + cell[0] - 2 * cell[1], cell, 2)  # one lattice vector off: the same regions
    assert torch.equal(idx, region_index(frac @ cell, cell, 2))
    assert torch.equal(idx, ((frac * 2).long() * torch.tensor([4, 2, 1])).sum(1))


def test_region_factor():
    assert math.isclose(region_factor(128), 1.0 + 4.0 / math.sqrt(768.0))
    assert torch.allclose(region_factor(torch.tensor([128.0, 6.0])), torch.tensor([region_factor(128), region_factor(6)]))


def test_md_backend_reports_the_worst_region_and_the_local_clause_refuses_it():
    # a residual that changes only on 16 atoms of one region: the whole-cell RMS dilutes it 32-fold in square,
    # the region 4-fold, so an epsilon between the two passes the magnitude clause and fails the local one
    import numpy as np

    from frost.integrators import Integrator, MDState
    from frost.runtime import LocalMagnitude, Magnitude
    from frost.runtime.md import MDBackend

    n_side, a = 8, 3.0
    r = torch.arange(n_side, dtype=torch.float64)
    sites = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3) * a + 0.5 * a
    cell = np.eye(3) * n_side * a
    hot = torch.zeros(len(sites), dtype=torch.bool)
    hot[torch.nonzero(region_index(sites, torch.as_tensor(cell), 2) == 0).squeeze(1)[:16]] = True
    c, w = 1.0, 2.0

    def student(x):
        d = x - sites
        return 0.5 * float((d ** 2).sum()), -d, x

    def teacher(x):
        e, f, _ = student(x)
        f = f.clone()
        f[hot] += c * torch.sin(w * x[hot])
        return e + float((c / w * torch.cos(w * x[hot])).sum()), f, x

    masses = torch.full((len(sites), 1), 10.0, dtype=torch.float64)
    v = torch.randn(sites.shape, generator=torch.Generator().manual_seed(1), dtype=torch.float64) * (8.617e-5 * 300 / 10.0) ** 0.5
    state = MDState(sites.clone(), v, masses)
    backend = MDBackend(state, Integrator(state, 1.0), student, teacher, 1.0, len(sites),
                        local={"regions": 2, "z": 4.0, "cell": cell})
    iv = backend.interval(4, "sync")
    assert iv.extra["local"] > 2.0 * iv.staleness
    assert abs(iv.extra["change"] - iv.extra["local"] * 3 ** 0.5) < 1e-12  # the controller grows on the binding one
    eps = (iv.extra["local"] * iv.staleness) ** 0.5
    assert Magnitude(epsilon=eps).check(iv) is None
    assert LocalMagnitude(epsilon=eps).check(iv) is not None
    assert LocalMagnitude(epsilon=eps).check(type(iv)(t=0.0, k=4, staleness=1.0)) is None  # no local measure: silent
