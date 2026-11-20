"""PBC edge identity and displacement continuity.

Guards failure mode #1: a wrapped coordinate makes an edge look like it moved
a box length, which inflates the dirty fraction with no visible symptom.
"""

import numpy as np

from frost.edgekey import ImageTracker, edge_keys, edge_vectors, minimum_image

CELL = np.eye(3) * 10.0


def test_minimum_image_ase_convention():
    dr = np.array([[9.0, 0.0, 0.0]])
    wrapped, S = minimum_image(dr, CELL)
    np.testing.assert_allclose(wrapped[0], [-1.0, 0.0, 0.0])
    np.testing.assert_array_equal(S[0], [-1, 0, 0])
    np.testing.assert_allclose(wrapped, dr + S @ CELL)


def test_wrapped_positions_produce_the_bug():
    """Document the failure explicitly, so a regression is unambiguous."""
    pos = np.array([[0.5, 5.0, 5.0], [9.5, 5.0, 5.0]])
    i, j = np.array([0]), np.array([1])
    _, S = minimum_image(pos[j] - pos[i], CELL)
    wrapped = pos.copy()
    wrapped[1, 0] = 0.1  # crossed the face, engine wrapped
    bad = np.linalg.norm(edge_vectors(wrapped, i, j, S, CELL)[0])
    assert bad > 9.0, "expected the spurious box-length jump"


def test_image_tracker_fixes_it():
    pos = np.array([[0.5, 5.0, 5.0], [9.5, 5.0, 5.0]])
    i, j = np.array([0]), np.array([1])
    _, S = minimum_image(pos[j] - pos[i], CELL)
    tr = ImageTracker(pos, CELL)
    tr.update(pos)
    wrapped = pos.copy()
    wrapped[1, 0] = 0.1
    good = np.linalg.norm(edge_vectors(tr.update(wrapped), i, j, S, CELL)[0])
    assert np.isclose(good, 0.4), f"expected 0.4 A, got {good}"


def test_keys_separate_periodic_images():
    i, j = np.array([0]), np.array([1])
    a = edge_keys(i, j, np.array([[-1, 0, 0]]))[0]
    b = edge_keys(i, j, np.array([[0, 0, 0]]))[0]
    assert a != b


def test_keys_unique_on_a_real_cell():
    from ase.neighborlist import neighbor_list

    from frost.systems import si512

    at = si512()
    i, j, S = neighbor_list("ijS", at, 6.0)
    k = edge_keys(i, j, S)
    assert len(np.unique(k)) == len(k)
    d = np.linalg.norm(
        edge_vectors(at.get_positions(), i, j, S, np.array(at.cell)), axis=1
    )
    assert d.max() <= 6.0 + 1e-6


def test_displacement_stays_physical_over_a_trajectory():
    from ase.neighborlist import neighbor_list

    from frost.systems import si512

    at = si512()
    i, j, S = neighbor_list("ijS", at, 6.0)
    c = np.array(at.cell)
    p = at.get_positions()
    tr = ImageTracker(p, c)
    tr.update(p)
    d0 = edge_vectors(p, i, j, S, c)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        p = p + rng.normal(0, 5e-3, p.shape)
        w = p - np.floor(p @ np.linalg.inv(c)) @ c
        worst = max(worst, np.abs(edge_vectors(tr.update(w), i, j, S, c) - d0).max())
    assert worst < 0.5, f"tracked displacement drifted to {worst} A"
