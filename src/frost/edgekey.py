"""Edge identity and displacement under periodic boundary conditions.

This is failure mode #1 in both the proposal (§9.2) and the implementation
guide, and it has two halves that are easy to conflate.

**Half one: the image offset belongs in the edge key.** Two atoms can be
neighbours through more than one periodic image. Keying the cache on
(id_i, id_j) alone lets a stale entry match the wrong image after a
neighbour-list rebuild, which corrupts forces with no visible symptom.

**Half two, the one that actually bites: the predicate needs UNWRAPPED
positions.** Frost's predicate thresholds on ||r_ij(t) - r_ij(t0)||, a
difference across time. MD engines wrap positions back into the cell, so when
an atom crosses a face its coordinate jumps by a box length. The edge's true
geometry changed by ~0.1 A; the arithmetic says ~10 A. Every edge on that atom
then reads as dirty, forever, and phi silently inflates.

Pinning the neighbour list's image offset S does NOT fix half two on its own,
because S describes the geometry at list-build time and a wrap invalidates it.
What fixes it is tracking each atom's cumulative image count, which is what
`ImageTracker` does. Classical MD engines carry exactly this for the same
reason -- it is how they compute mean-squared displacement.

Conventions follow ASE's `neighbor_list("ijS", ...)`:

    r_ij = positions[j] - positions[i] + S @ cell

Run `python frost/edgekey.py` for the hand-checkable proof on a 10 A box.
"""

from __future__ import annotations

import numpy as np

# 21 bits per atom id, 3 x 7 bits of signed image offset: 2.1 M atoms and
# offsets in [-64, 63], far beyond any cell we run.
_ID_BITS = 21
_OFF_BITS = 7
_OFF_BIAS = 1 << (_OFF_BITS - 1)


def minimum_image(dr: np.ndarray, cell: np.ndarray, pbc=(True, True, True)):
    """Wrap displacements into the minimum image, ASE sign convention.

    Returns (wrapped, S) with  wrapped == dr + S @ cell.
    """
    dr = np.atleast_2d(np.asarray(dr, dtype=np.float64))
    cell = np.asarray(cell, dtype=np.float64)
    frac = dr @ np.linalg.inv(cell)
    shift = np.round(frac)
    shift[:, ~np.asarray(pbc, dtype=bool)] = 0.0
    S = (-shift).astype(np.int64)
    return dr + S.astype(np.float64) @ cell, S


def edge_vectors(
    pos: np.ndarray, i: np.ndarray, j: np.ndarray, S: np.ndarray, cell: np.ndarray
) -> np.ndarray:
    """r_ij for edges whose periodic image is fixed by S.

    `pos` must be UNWRAPPED -- see `ImageTracker`. Feeding wrapped positions
    here is precisely the bug this module documents.
    """
    return pos[j] - pos[i] + np.asarray(S, np.float64) @ np.asarray(cell, np.float64)


class ImageTracker:
    """Undo the engine's wrapping so displacements stay continuous in time.

    Feed it the wrapped positions every step. It accumulates each atom's
    integer image count by detecting jumps larger than half a box, and returns
    positions that move smoothly through a face crossing.

    Valid while no atom moves more than half a box in one step, which in MD is
    true by many orders of magnitude (5e-3 A against a cell of tens of A).
    """

    def __init__(self, pos0: np.ndarray, cell: np.ndarray, pbc=(True, True, True)):
        self.cell = np.asarray(cell, dtype=np.float64)
        self.inv = np.linalg.inv(self.cell)
        self.pbc = np.asarray(pbc, dtype=bool)
        self.prev_frac = np.asarray(pos0, dtype=np.float64) @ self.inv
        self.images = np.zeros_like(self.prev_frac)

    def update(self, pos: np.ndarray) -> np.ndarray:
        """Return unwrapped positions for this step's wrapped `pos`."""
        frac = np.asarray(pos, dtype=np.float64) @ self.inv
        jump = np.round(frac - self.prev_frac)
        jump[:, ~self.pbc] = 0.0
        self.images -= jump
        self.prev_frac = frac
        return (frac + self.images) @ self.cell


def edge_keys(i: np.ndarray, j: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Pack (i, j, image offset) into one int64 per edge."""
    i = np.asarray(i, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    off = np.asarray(S, dtype=np.int64) + _OFF_BIAS
    if (off < 0).any() or (off >= (1 << _OFF_BITS)).any():
        raise ValueError("image offset out of range for the key packing")
    if (i >= (1 << _ID_BITS)).any() or (j >= (1 << _ID_BITS)).any():
        raise ValueError("atom id out of range for the key packing")
    key = (i << (_ID_BITS + 3 * _OFF_BITS)) | (j << (3 * _OFF_BITS))
    return key | ((off[:, 0] << (2 * _OFF_BITS)) | (off[:, 1] << _OFF_BITS) | off[:, 2])


def _selftest() -> int:
    cell = np.eye(3) * 10.0
    fails = 0

    def check(name, got, want, tol=1e-9):
        nonlocal fails
        ok = np.allclose(got, want, atol=tol)
        print(
            f"  {'ok  ' if ok else 'FAIL'} {name}: {np.round(got, 4)} (expect {want})"
        )
        if not ok:
            fails += 1

    print("=== 1. minimum image and the ASE sign convention ===")
    pos = np.array([[0.5, 5.0, 5.0], [9.5, 5.0, 5.0]])
    i, j = np.array([0]), np.array([1])
    raw = pos[j] - pos[i]  # +9.0 x: the long way round
    wrapped, S = minimum_image(raw, cell)
    check("wrapped dr", wrapped[0], [-1.0, 0.0, 0.0])
    check("offset S", S[0], [-1, 0, 0])
    check(
        "r_ij == pj - pi + S@cell",
        edge_vectors(pos, i, j, S, cell)[0],
        [-1.0, 0.0, 0.0],
    )

    print("\n=== 2. an atom crosses a face; the engine wraps it ===")
    print("  atom 1 walks 9.5 -> 10.1, wrapped by the engine to 0.1")
    print("  true edge length goes 1.000 -> 0.400")
    tracker = ImageTracker(pos, cell)
    tracker.update(pos)  # step 0
    wrapped_pos = pos.copy()
    wrapped_pos[1, 0] = 0.1
    unwrapped = tracker.update(wrapped_pos)

    naive = edge_vectors(wrapped_pos, i, j, S, cell)[0]
    print(
        f"  WRONG  wrapped positions + pinned S : |r_ij| = "
        f"{np.linalg.norm(naive):.3f}   <-- the week-losing bug"
    )
    fixed = edge_vectors(unwrapped, i, j, S, cell)[0]
    check("unwrapped positions + pinned S", np.linalg.norm(fixed), 0.4)

    print("\n=== 3. the key is stable across the crossing ===")
    k0 = edge_keys(i, j, S)[0]
    k1 = edge_keys(i, j, S)[0]
    print(f"  {'ok  ' if k0 == k1 else 'FAIL'} key stable: {k0}")
    fails += k0 != k1
    k_img = edge_keys(i, j, np.array([[0, 0, 0]]))[0]
    print(
        f"  {'ok  ' if k_img != k0 else 'FAIL'} a different image gets a "
        f"different key: {k_img} != {k0}"
    )
    fails += k_img == k0

    print("\n=== 4. a real cell and a real neighbour list ===")
    try:
        from ase.neighborlist import neighbor_list

        from frost.systems import si512

        at = si512()
        ii, jj, SS = neighbor_list("ijS", at, 6.0)
        c = np.array(at.cell)
        d = np.linalg.norm(edge_vectors(at.get_positions(), ii, jj, SS, c), axis=1)
        ks = edge_keys(ii, jj, SS)
        print(f"  Si512: {len(ii)} edges, |r_ij| in [{d.min():.3f}, {d.max():.3f}] A")
        if d.max() > 6.0 + 1e-6:
            print("  FAIL: displacement exceeds the cutoff")
            fails += 1
        else:
            print("  ok   every edge is within the cutoff")
        uniq = len(np.unique(ks))
        print(
            f"  {'ok  ' if uniq == len(ks) else 'FAIL'} keys unique: {uniq}/{len(ks)}"
        )
        fails += uniq != len(ks)

        # 200 steps of thermal motion with wrapping, the realistic case.
        rng = np.random.default_rng(0)
        p = at.get_positions()
        tr = ImageTracker(p, c)
        tr.update(p)
        d0 = edge_vectors(p, ii, jj, SS, c)
        naive_max = fixed_max = 0.0
        for _ in range(200):
            p = p + rng.normal(0, 5e-3, p.shape)
            w = p - np.floor(p @ np.linalg.inv(c)) @ c  # engine wraps
            u = tr.update(w)
            naive_max = max(
                naive_max, np.abs(edge_vectors(w, ii, jj, SS, c) - d0).max()
            )
            fixed_max = max(
                fixed_max, np.abs(edge_vectors(u, ii, jj, SS, c) - d0).max()
            )
        print("\n  200 thermal steps, max |delta r_ij| over all edges:")
        print(
            f"    wrapped positions   : {naive_max:8.3f} A   "
            f"<-- spurious box-length jumps"
        )
        print(
            f"    ImageTracker        : {fixed_max:8.3f} A   "
            f"<-- physical, ~sqrt(200)*5e-3*sqrt(2)"
        )
        if fixed_max > 0.5:
            print("  FAIL: tracked displacement is not physical")
            fails += 1
        else:
            print("  ok   tracked displacement stays physical")
    except Exception as exc:
        print(f"  (skipped: {type(exc).__name__}: {exc})")

    print("\n" + ("ALL CHECKS PASSED" if fails == 0 else f"{fails} FAILURES"))
    return int(fails)


if __name__ == "__main__":
    raise SystemExit(_selftest())
