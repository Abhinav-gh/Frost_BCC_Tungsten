"""The alanine-dipeptide rate observable (experiments/e13_runtime/dihedrals.py): angles, cores, transition counting."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "e13_runtime"))

from dihedrals import analyse_angles, committed, core_of, dihedral

from frost.systems import ALA2_PHI, ALA2_PSI, ala2_water


def test_dihedral_matches_ase_and_survives_wrapping():
    at = ala2_water()
    box = np.diag(at.cell.array)
    x = at.get_positions()[None]
    for idx in (ALA2_PHI, ALA2_PSI):
        ref = (at.get_dihedral(*idx, mic=True) + 180) % 360 - 180
        assert abs(dihedral(x, idx, box)[0] - ref) < 1e-6
        # shifting one atom by a box vector (an unwrapped coordinate) changes nothing
        y = x.copy()
        y[0, idx[0], 0] += box[0]
        assert abs(dihedral(y, idx, box)[0] - ref) < 1e-6
    assert abs(dihedral(x, ALA2_PHI, box)[0] - (-80.6)) < 0.2  # the C7eq start


def test_cores_and_committed_assignment():
    phi = np.array([-80, -80, -80, -70, -70, -80, 60])
    psi = np.array([150, 60, -40, 60, 140, -170, 30])
    core = core_of(phi, psi)
    assert core.tolist() == [0, -1, 1, -1, 0, 0, 2]
    assert committed(core).tolist() == [0, 0, 1, 1, 0, 0, 2]
    assert committed(np.array([-1, -1, 1])).tolist() == [-1, -1, 1]


def test_transitions_ignore_recrossings_between_cores():
    # beta -> (gap, back to beta side) -> beta -> alphaR -> gap -> alphaR: one transition only
    phi = np.full(8, -80.0)
    psi = np.array([150, 80, 150, 150, -40, 50, -40, -40], dtype=float)
    r = analyse_angles(phi, psi, 20.0, 300.0)
    assert r["n_transitions"] == 1 and r["transitions"] == {"beta->alphaR": 1}
