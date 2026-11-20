"""Constant pressure pieces (further_expansions_beyond.md 6.1): a list follows a cell move exactly, and the stochastic
cell-rescaling barostat moves the volume toward the target pressure."""

from __future__ import annotations

import ase.build
import numpy as np
import torch

from frost.integrators import StochasticCellRescaling
from frost.neighbours import VerletList


def test_list_follows_a_cell_move_exactly():
    atoms = ase.build.bulk("Cu", "fcc", a=3.61, cubic=True).repeat((4, 4, 4))
    x = torch.as_tensor(atoms.get_positions(), dtype=torch.float64) + 0.05 * torch.randn(256, 3, dtype=torch.float64)
    lst = VerletList(atoms.cell.array, 4.5, 1.0, "cpu", wrap_positions=False, builder="vesin")
    lst.update(x, force=True)
    r_before = x[lst.j] - x[lst.i] + lst.shifts
    mu = 1.003
    lst.set_cell(atoms.cell.array * mu)
    xs = x * mu
    r_after = xs[lst.j] - xs[lst.i] + lst.shifts
    assert torch.allclose(r_after, mu * r_before, atol=1e-12)  # every edge vector scales by mu: a uniform expansion


def test_barostat_moves_the_volume_toward_the_target_pressure():
    from ase import units

    baro = StochasticCellRescaling(pressure_bar=1.0, tau_fs=1000.0, compressibility_per_bar=1e-5, temperature_K=300.0)
    baro.kT = 0.0  # noise and the k_B T / V term off: the deterministic (Berendsen) limit
    v = 1.0e4
    high = baro.pressure(kinetic=0.0, virial=3.0 * v * 1000.0 * units.bar, volume=v)  # 1,000 bar from the virial
    assert abs(high / units.bar - 1000.0) < 1e-6
    assert baro.scale(v, high, 10 * units.fs) > 1.0  # too high a pressure: the box expands
    assert baro.scale(v, -500.0 * units.bar, 10 * units.fs) < 1.0  # tension: it shrinks
    assert abs(baro.scale(v, baro.p0, 10 * units.fs) - 1.0) < 1e-15  # at the target: no move


def test_barostat_noise_has_the_bernetti_bussi_size():
    from ase import units

    baro = StochasticCellRescaling(pressure_bar=1.0, tau_fs=1000.0, compressibility_per_bar=1e-5, temperature_K=300.0)
    v, dt = 1.0e4, 10 * units.fs
    deps = np.array([3.0 * np.log(baro.scale(v, baro.p0 + baro.kT / v, dt)) for _ in range(4000)])
    expected = np.sqrt(2.0 * baro.kT * baro.beta * dt / (v * baro.tau))
    assert abs(deps.std() / expected - 1.0) < 0.05 and abs(deps.mean()) < 4 * expected / np.sqrt(4000)


def _lj_system(n_side=3, a=5.26, seed=0):
    """An argon-like Lennard-Jones fcc crystal, brute-force minimum image (CPU, tiny)."""
    atoms = ase.build.bulk("Ar", "fcc", a=a, cubic=True).repeat((n_side, n_side, n_side))
    box = {"cell": np.array(atoms.cell.array, dtype=np.float64)}
    eps, sig, rc = 0.0104, 3.40, 7.0  # rc under half the 15.8 A box (minimum image stays one image per pair)
    e_rc = 4 * eps * ((sig / rc) ** 12 - (sig / rc) ** 6)  # shifted to zero at rc: no energy jump when a pair crosses it

    def pairs(x):
        c = torch.as_tensor(box["cell"], dtype=torch.float64)
        d = x[None, :, :] - x[:, None, :]
        frac = d @ torch.linalg.inv(c)
        d = (frac - torch.round(frac)) @ c
        r = d.norm(dim=-1)
        mask = (r > 0) & (r < rc)
        return d, r, mask

    def force(x):
        d, r, mask = pairs(x)
        rs = torch.where(mask, r, torch.ones_like(r))
        sr6 = (sig / rs) ** 6
        e = torch.where(mask, 4 * eps * (sr6 * sr6 - sr6) - e_rc, torch.zeros_like(r)).sum() / 2
        de = torch.where(mask, 4 * eps * (-12 * sr6 * sr6 + 6 * sr6) / rs, torch.zeros_like(r))  # dE/dr
        f = (de / rs)[:, :, None] * d  # on atom i from j: -dE/dr * (r_i - r_j)/r = dE/dr * d/r with d = r_j - r_i
        return e, f.sum(1), x

    def virial():
        x = state_box["x"]
        _, r, mask = pairs(x)
        rs = torch.where(mask, r, torch.ones_like(r))
        sr6 = (sig / rs) ** 6
        de = torch.where(mask, 4 * eps * (-12 * sr6 * sr6 + 6 * sr6) / rs, torch.zeros_like(r))
        return float(-(rs * de).sum() / 2)  # W = -sum over pairs of r dE/dr

    def set_cell(c):
        box["cell"] = np.array(c, dtype=np.float64)

    state_box = {}
    x = torch.as_tensor(atoms.get_positions(), dtype=torch.float64)
    masses = torch.full((len(atoms), 1), 39.948, dtype=torch.float64)
    return atoms, box, x, masses, force, virial, set_cell, state_box


def test_barostat_work_is_booked_so_energy_minus_heat_stays_flat():
    from frost.integrators import Integrator, MDState
    from frost.runtime.md import MDBackend

    atoms, box, x, masses, force, virial, set_cell, state_box = _lj_system()
    g = torch.Generator().manual_seed(0)
    v = torch.randn(x.shape, generator=g, dtype=torch.float64) * torch.sqrt(8.617e-5 * 60.0 / masses)
    state = MDState(x, v, masses)
    integ = Integrator(state, 2.0)
    integ.track_heat()
    state_box["x"] = state.x

    def tracked(xx):
        state_box["x"] = xx
        return force(xx)

    baro = StochasticCellRescaling(pressure_bar=1.0, tau_fs=300.0, compressibility_per_bar=1e-4, temperature_K=60.0, seed=3)
    backend = MDBackend(state, integ, tracked, tracked, 2.0, len(atoms),
                        barostat={"baro": baro, "virial": virial, "set_cell": set_cell, "cell": box["cell"]})
    raw, eff = [], []
    for _ in range(150):
        iv = backend.interval(4, "sync")
        eff.append(iv.energy)  # (E - Q - E0) / N in meV/atom
        raw.append(iv.energy + 1e3 * float(integ.heat) / len(atoms))
    vols = [v for _, v in backend.volume_trace]
    assert max(vols) / min(vols) > 1.002  # the box really moved
    assert (max(raw) - min(raw)) > 5 * (max(eff) - min(eff))  # the moves change E; booked as heat, E - Q stays flat
    assert max(eff) - min(eff) < 0.2  # meV/atom: integration error only


def test_uniaxial_barostat_moves_only_z_and_reads_the_normal_stress():
    from ase import units

    baro = StochasticCellRescaling(pressure_bar=1.0, tau_fs=1000.0, compressibility_per_bar=1e-5, temperature_K=300.0,
                                   axes="z")
    baro.kT = 0.0
    v = 1.0e4
    w = np.diag([0.0, 0.0, v * 2000.0 * units.bar])  # a 2,000 bar normal stress, nothing laterally
    p_zz = baro.pressure_axes([0.0, 0.0, 0.0], w, v)
    assert abs(p_zz / units.bar - 2000.0) < 1e-6
    mu = baro.scale_vector(v, p_zz, 10 * units.fs)
    assert mu[0] == 1.0 and mu[1] == 1.0 and mu[2] > 1.0  # only L_z expands
    iso = StochasticCellRescaling(pressure_bar=1.0, tau_fs=1000.0, compressibility_per_bar=1e-5, temperature_K=300.0)
    assert abs(iso.pressure_axes([1.0, 2.0, 3.0], np.eye(3), v) - iso.pressure(6.0, 3.0, v)) < 1e-18


def test_uniaxial_moves_keep_the_lateral_cell_and_book_heat():
    from frost.integrators import Integrator, MDState
    from frost.runtime.md import MDBackend

    atoms, box, x, masses, force, virial, set_cell, state_box = _lj_system()
    eps, sig, rc = 0.0104, 3.40, 7.0

    def virial_tensor():
        xx = state_box["x"]
        c = torch.as_tensor(box["cell"], dtype=torch.float64)
        d = xx[None, :, :] - xx[:, None, :]
        frac = d @ torch.linalg.inv(c)
        d = (frac - torch.round(frac)) @ c
        r = d.norm(dim=-1)
        mask = (r > 0) & (r < rc)
        rs = torch.where(mask, r, torch.ones_like(r))
        sr6 = (sig / rs) ** 6
        de = torch.where(mask, 4 * eps * (-12 * sr6 * sr6 + 6 * sr6) / rs, torch.zeros_like(r))
        return -((de / rs)[:, :, None, None] * d[:, :, :, None] * d[:, :, None, :]).sum((0, 1)) / 2

    g = torch.Generator().manual_seed(0)
    v = torch.randn(x.shape, generator=g, dtype=torch.float64) * torch.sqrt(8.617e-5 * 60.0 / masses)
    state = MDState(x, v, masses)
    integ = Integrator(state, 2.0)
    integ.track_heat()
    state_box["x"] = state.x
    assert abs(float(torch.trace(virial_tensor())) - virial()) < 1e-9 * abs(virial())

    def tracked(xx):
        state_box["x"] = xx
        return force(xx)

    cell0 = box["cell"].copy()
    baro = StochasticCellRescaling(pressure_bar=1.0, tau_fs=300.0, compressibility_per_bar=1e-4, temperature_K=60.0, seed=3,
                                   axes="z")
    backend = MDBackend(state, integ, tracked, tracked, 2.0, len(atoms),
                        barostat={"baro": baro, "virial": virial, "virial_tensor": virial_tensor, "set_cell": set_cell,
                                  "cell": box["cell"]})
    raw, eff = [], []
    for _ in range(150):
        iv = backend.interval(4, "sync")
        eff.append(iv.energy)
        raw.append(iv.energy + 1e3 * float(integ.heat) / len(atoms))
    assert np.array_equal(backend.cell[:, :2], cell0[:, :2])  # the lateral cell never moves
    zs = [v for _, v in backend.volume_trace]
    assert max(zs) / min(zs) > 1.002  # L_z really moved
    assert (max(raw) - min(raw)) > 5 * (max(eff) - min(eff))
    assert max(eff) - min(eff) < 0.2


def test_uniaxial_tension_stretches_x_at_the_set_rate_and_books_the_work():
    """Mechanics (further_expansions_beyond.md 7.5): x stretched at a constant true strain rate, y and z each under their
    own barostat; the cell's x length follows exp(rate t) exactly, the lateral axes move, E - Q stays flat (the
    deformation's work is booked like a barostat move), and the stress record fills."""
    import math

    from ase import units

    from frost.integrators import Integrator, MDState
    from frost.runtime.md import MDBackend

    atoms, box, x, masses, force, virial, set_cell, state_box = _lj_system()
    eps, sig, rc = 0.0104, 3.40, 7.0

    def virial_tensor():
        xx = state_box["x"]
        c = torch.as_tensor(box["cell"], dtype=torch.float64)
        d = xx[None, :, :] - xx[:, None, :]
        frac = d @ torch.linalg.inv(c)
        d = (frac - torch.round(frac)) @ c
        r = d.norm(dim=-1)
        mask = (r > 0) & (r < rc)
        rs = torch.where(mask, r, torch.ones_like(r))
        sr6 = (sig / rs) ** 6
        de = torch.where(mask, 4 * eps * (-12 * sr6 * sr6 + 6 * sr6) / rs, torch.zeros_like(r))
        return -((de / rs)[:, :, None, None] * d[:, :, :, None] * d[:, :, None, :]).sum((0, 1)) / 2

    g = torch.Generator().manual_seed(0)
    v = torch.randn(x.shape, generator=g, dtype=torch.float64) * torch.sqrt(8.617e-5 * 60.0 / masses)
    state = MDState(x, v, masses)
    integ = Integrator(state, 2.0)
    integ.track_heat()
    state_box["x"] = state.x

    def tracked(xx):
        state_box["x"] = xx
        return force(xx)

    rate = 2e-5  # per fs (2e10 / s): 1.2% over the 600 fs below
    cell0 = box["cell"].copy()
    baro = StochasticCellRescaling(pressure_bar=1.0, tau_fs=300.0, compressibility_per_bar=1e-4, temperature_K=60.0,
                                   seed=4, axes="yz", deform=("x", rate))
    backend = MDBackend(state, integ, tracked, tracked, 2.0, len(atoms),
                        barostat={"baro": baro, "virial": virial, "virial_tensor": virial_tensor, "set_cell": set_cell,
                                  "cell": box["cell"]})
    eff = []
    for _ in range(75):
        iv = backend.interval(4, "sync")
        eff.append(iv.energy)
    t_moves = 75 * 4 * 2.0  # one move per interval, at its start, each lasting the interval
    assert abs(backend.cell[0, 0] / cell0[0, 0] - math.exp(rate * t_moves)) < 1e-12
    assert backend.cell[1, 1] != cell0[1, 1] and backend.cell[2, 2] != cell0[2, 2]
    assert max(eff) - min(eff) < 0.2
    assert len(backend.stress_trace) >= 70 and len(backend.stress_trace[0]) == 7
    assert units.bar > 0
