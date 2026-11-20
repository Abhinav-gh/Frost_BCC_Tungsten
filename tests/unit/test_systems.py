"""Every simulation cell must be physically sane before it reaches a model.

This exists because it did not, once. The silica cell was built by averaging
fractional coordinates without minimum-image correction, which put several
oxygens on top of each other at a separation of exactly zero. The forces were
not finite, and two MD runs spun for nineteen hours without completing a single
logged step before anyone noticed, because a stuck run and a slow run look
identical from outside.

A cell with coincident atoms is the cheapest possible thing to detect and one
of the most expensive to detect late, so it is checked here rather than
discovered in a queue.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase.neighborlist import neighbor_list

from frost.systems import E1_SYSTEMS, GAS_SYSTEMS, si_scaling

# Below this, two atoms are on top of each other by any chemical standard.
# The shortest real bond in the E1 set is the O-H of water at 0.97 A, so the
# global floor has to sit under that; the silica failure this guards against
# was exactly 0.0000 A. Contacts that are close but not bonded are checked
# per-system instead, where the chemistry is known.
MIN_SEPARATION = 0.8


def _cells():
    for name, fn in E1_SYSTEMS.items():
        try:
            yield name, fn()
        except RuntimeError as exc:  # li3po4 without a cached CIF or API key
            pytest.skip(f"{name}: {exc}")


@pytest.mark.parametrize("name", list(E1_SYSTEMS))
def test_no_coincident_atoms(name):
    fn = E1_SYSTEMS[name]
    try:
        at = fn()
    except RuntimeError as exc:
        pytest.skip(f"{name}: {exc}")
    # 5 A, not 3: alkali metals and Mg are condensed with nearest neighbours at 3.0-4.6 A (Li 3.04, Mg 3.20, Na 3.72,
    # K 4.61); the coincidence floor below is what this test guards, density is checked separately.
    d = neighbor_list("d", at, 5.0)
    assert d.size, f"{name}: no neighbours within 5 A" + ("" if name in GAS_SYSTEMS else ", the cell is not condensed")
    assert d.min() > MIN_SEPARATION, (
        f"{name}: minimum interatomic distance is {d.min():.4f} A. "
        "Atoms are coincident or nearly so; forces will not be finite."
    )


@pytest.mark.parametrize("name", [n for n in E1_SYSTEMS if n not in GAS_SYSTEMS])
def test_density_and_degree_are_physical(name):
    fn = E1_SYSTEMS[name]
    try:
        at = fn()
    except RuntimeError as exc:
        pytest.skip(f"{name}: {exc}")
    rho = len(at) / at.get_volume()
    assert 0.01 < rho < 0.35, (
        f"{name}: number density {rho:.4f} per A^3 is not condensed"
    )

    i, _ = neighbor_list("ij", at, 6.0)
    degree = len(i) / len(at)
    # MACE-MP-0 runs at r_c = 6 A, where a condensed phase gives tens of
    # neighbours. A degree near zero means the cell is too small for the cutoff.
    assert 10 < degree < 200, f"{name}: mean degree {degree:.1f} at r_c = 6 A"


@pytest.mark.parametrize("name", list(E1_SYSTEMS))
def test_cell_is_periodic_and_finite(name):
    fn = E1_SYSTEMS[name]
    try:
        at = fn()
    except RuntimeError as exc:
        pytest.skip(f"{name}: {exc}")
    assert at.pbc.all(), f"{name}: E1 runs condensed phases; all axes must be periodic"
    assert np.isfinite(at.get_positions()).all()
    assert at.get_volume() > 0


def test_silica_is_stoichiometric_cristobalite():
    """Si-O bonding, not just a count of atoms.

    Beta-cristobalite has every silicon four-coordinate and every oxygen
    bridging exactly two silicons. Both numbers come out wrong the moment the
    bridging oxygens are misplaced, which is the failure this guards.
    """
    at = E1_SYSTEMS["silica1152"]()
    sym = np.array(at.get_chemical_symbols())
    n_si = int((sym == "Si").sum())
    n_o = int((sym == "O").sum())
    assert n_o == 2 * n_si, f"SiO2 needs 2:1, got Si{n_si}O{n_o}"

    i, _, d = neighbor_list("ijd", at, 2.0)
    cn = np.bincount(i[d < 2.0], minlength=len(at))
    assert set(np.unique(cn[sym == "Si"])) == {4}, "silicon must be 4-coordinate"
    assert set(np.unique(cn[sym == "O"])) == {2}, "oxygen must bridge two silicons"


def test_water_has_no_intermolecular_overlap():
    """A bonded O-H at 0.97 A is fine; two molecules at 0.85 A are not.

    The global minimum cannot tell those apart, so water is checked on
    intermolecular contacts only. Equilibration does anneal an overlap away,
    but relying on that makes the starting structure seed-dependent.
    """
    at = E1_SYSTEMS["water216"]()
    i, j, d = neighbor_list("ijd", at, 2.2)
    molecule_of = np.arange(len(at)) // 3  # the builder emits O, H, H
    inter = molecule_of[i] != molecule_of[j]
    assert inter.any(), "no intermolecular neighbours; the box is not condensed"
    # 1.6 A is below the O...H hydrogen bond at about 1.8 A and well above any
    # repulsive contact.
    assert d[inter].min() > 1.6, (
        f"closest intermolecular contact is {d[inter].min():.3f} A; molecules overlap"
    )


def test_scaling_cells_carry_their_defect():
    """E26 sweeps system size at fixed defect structure; the defect must exist."""
    bulk_cell = si_scaling(3)
    vac = si_scaling(3, defect="vacancy")
    surf = si_scaling(3, defect="surface")
    assert len(vac) == len(bulk_cell) - 1, "the vacancy removed no atom"
    assert surf.cell.array[2, 2] > bulk_cell.cell.array[2, 2], "no vacuum added"
    for at in (bulk_cell, vac, surf):
        d = neighbor_list("d", at, 3.0)
        assert d.min() > MIN_SEPARATION


def test_screw_dipole_has_opposite_burgers_vectors_a_periodic_field_and_no_overlaps():
    """bcc_screw_dipole (2026-11-18): the circuit around each core closes with +-b and around both with 0; the field
    jumps by the same amount (mod b) across a cell everywhere, which the cell tilt takes; no two atoms closer than 2.2 A."""
    from frost.neighbours import edges_for
    from frost.systems import bcc_screw_dipole, screw_dipole_field

    at = bcc_screw_dipole("W", nx=6, ny=10, nz=2, images=20)
    assert len(at) == 6 * 6 * 10 * 2
    b = at.info["burgers"]
    lx, ly = np.linalg.norm(at.cell.array[0, :2]), np.linalg.norm(at.cell.array[1, :2])
    cores = at.info["screw_cores"]

    def winding(centre, radius):
        phi = np.linspace(0.0, 2.0 * np.pi, 721)
        loop = np.stack([centre[0] + radius * np.cos(phi), centre[1] + radius * np.sin(phi)], 1)
        u = screw_dipole_field(loop, cores, b, lx, ly, 20)
        d = np.diff(u)
        d = (d + b / 2) % b - b / 2  # each small step's change, the +-b branch jumps removed
        return float(d.sum())

    (xa, ya, _), (xb, yb, _) = cores
    assert abs(winding((xa, ya), 0.8) - b) < 1e-6
    assert abs(winding((xb, yb), 0.8) + b) < 1e-6
    big = 0.5 * np.hypot(xb - xa, yb - ya) + 3.0
    assert abs(winding(((xa + xb) / 2, (ya + yb) / 2), big)) < 1e-6
    rng = np.random.default_rng(1)
    probe = rng.random((200, 2)) * [lx, ly]
    for shift in ((lx, 0.0), (0.0, ly)):
        jump = (screw_dipole_field(probe + shift, cores, b, lx, ly, 20)
                - screw_dipole_field(probe, cores, b, lx, ly, 20)) % b
        assert float(jump.std()) < 0.02 * b
    _, i, _, _, _ = edges_for(at.get_positions(), at.cell.array, 2.2)
    assert int(i.numel()) == 0


def test_stress_free_screw_dipole_is_periodic_with_its_plastic_tilts():
    """bcc_screw_dipole(stress_free=True) (2026-11-18): the image sum's linear term out, the dipole's plastic distortion in;
    the field's jump across one cell equals the cell's tilt there, modulo b, everywhere (so the tilted cell is periodic),
    and the tilts are the plastic values b dy / ly and -b dx / lx."""
    from frost.systems import bcc_screw_dipole, screw_dipole_field

    at = bcc_screw_dipole("W", nx=6, ny=10, nz=2, images=20, stress_free=True)
    cell = at.cell.array
    b, cores = at.info["burgers"], at.info["screw_cores"]
    lx, ly = cell[0, 0], cell[1, 1]
    (xa, ya, _), (xb, yb, _) = cores
    assert abs(cell[0, 2] - b * (yb - ya) / ly) < 1e-9 and abs(cell[1, 2] + b * (xb - xa) / lx) < 1e-9
    (l0, l1), (p0, p1) = at.info["linear_term"], at.info["plastic_tilts"]

    def z(p):
        return (screw_dipole_field(p, cores, b, lx, ly, 20) + (p0 - l0) * p[:, 0] / lx
                + (p1 - l1) * p[:, 1] / ly)

    rng = np.random.default_rng(2)
    probe = rng.random((300, 2)) * [lx, ly]
    for shift, tilt in (((lx, 0.0), cell[0, 2]), ((0.0, ly), cell[1, 2])):
        off = z(probe + np.array(shift)) - z(probe) - tilt
        off = (off + b / 2) % b - b / 2
        assert float(np.abs(off).max()) < 0.02 * b
