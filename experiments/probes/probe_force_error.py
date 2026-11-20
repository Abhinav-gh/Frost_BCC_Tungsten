"""Test Theorem 6.1 and 6.2 directly on MACE-MP-0.

The proposal's sharpest claim is that a zeroth-order edge cache does not
*approximate* the force, it *deletes* the force contribution of every cached
edge, with an error of order ||J|| that does not shrink as the tolerance
shrinks.  A first-order cache repairs this, with error linear in displacement.

This is the check the implementation guide says gates everything downstream:
"If the zeroth-order error is not flat in eps, stop and debug."

Method.  Hook interaction 0's `conv_tp` and substitute its output.  The module
is trilinear in its three inputs -- node features, spherical harmonics, radial
weights -- so the first-order cache can be written exactly without ever
materialising the Jacobian:

    g(a, b, w)  =  TP(a, b, w)                                 exact
    g_0         =  TP(a0, b0, w0)                              zeroth order
    g_1         =  g_0 + TP(da, b0, w0)
                       + TP(a0, db, w0) + TP(a0, b0, dw)       first order

with d* = current - reference.  The reference terms are detached constants; the
deltas stay attached, which is what carries the gradient.  At layer 1, `a` is
the species embedding and is constant under the dynamics, so `da` vanishes --
that is Proposition 3.4 reappearing as an algebraic fact.

The decisive point in the sweep is displacement zero.  There the first-order
cache must be exact and the zeroth-order cache must already be badly wrong,
because the defect is a missing derivative, not a stale value.

Run:  source scripts/env.sh mace && python e1/probe_force_error.py
"""

from __future__ import annotations

import numpy as np
import torch
from ase.build import bulk

SITE = "interactions.0.conv_tp"


class NeighbourListChangedError(RuntimeError):
    """Raised when an edge crosses the cutoff and the edge set is no longer
    comparable to the reference. Frost handles this with generation counters;
    this probe simply skips the affected point."""

    def __init__(self, now: int, ref: int):
        super().__init__(f"edge count {ref} -> {now}")
        self.now, self.ref = now, ref


class EdgeCacheSubstituter:
    """Replace one tensor product's output with a cached surrogate."""

    def __init__(self, module):
        self.module = module
        self.mode = "record"  # record | exact | zeroth | first
        self.ref = None  # (a0, b0, w0, g0), all detached
        self._busy = False
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, mod, inputs, output):
        if self._busy:
            return None
        a, b, w = inputs[0], inputs[1], inputs[2]

        if self.mode == "record":
            self.ref = (
                a.detach().clone(),
                b.detach().clone(),
                w.detach().clone(),
                output.detach().clone(),
            )
            return None
        if self.mode == "exact" or self.ref is None:
            return None

        a0, b0, w0, g0 = self.ref
        if a.shape != a0.shape:
            # The neighbour list rebuilt: an edge crossed the cutoff. In Frost
            # proper this is a cache-invalidation event handled by the
            # generation counter; here we just refuse the comparison.
            raise NeighbourListChangedError(a.shape[0], a0.shape[0])

        if self.mode == "zeroth":
            # A detached constant: d(g)/d(r) is identically zero. Theorem 6.1.
            return g0

        if self.mode == "first":
            da, db, dw = a - a0, b - b0, w - w0
            self._busy = True
            try:
                corr = (
                    self.module(da, b0, w0)
                    + self.module(a0, db, w0)
                    + self.module(a0, b0, dw)
                )
            finally:
                self._busy = False
            return g0 + corr

        raise ValueError(self.mode)

    def close(self):
        self.handle.remove()


def forces(calc, atoms) -> np.ndarray:
    """Force a genuine re-evaluation.

    ASE calculators cache `results` and skip recomputation when the atoms are
    unchanged.  We call it three times on identical positions with different
    substitution modes, so the cache has to be cleared or every mode returns
    the first mode's answer -- silently, as all-zero error.
    """
    calc.reset()
    a = atoms.copy()
    a.calc = calc
    return a.get_forces()


def main() -> int:
    from mace.calculators import mace_mp

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    calc = mace_mp(model="medium", device=dev, default_dtype="float64")
    module = dict(calc.models[0].named_modules())[SITE]
    sub = EdgeCacheSubstituter(module)

    # Small cell so the neighbour list is stable under the displacements we
    # apply, and so the sweep is quick.
    ref_atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
    rng = np.random.default_rng(0)
    # Start from a thermally rattled configuration: a perfect crystal has zero
    # force by symmetry, which would make every relative error meaningless.
    ref_atoms.set_positions(
        ref_atoms.get_positions() + rng.normal(0, 0.05, ref_atoms.positions.shape)
    )
    ref_pos = ref_atoms.get_positions()

    print(f"system: {len(ref_atoms)} atoms of Si, device {dev}")

    sub.mode = "record"
    f_ref = forces(calc, ref_atoms)
    n_edges = sub.ref[0].shape[0]
    print(f"edges : {n_edges}  (mean degree {n_edges / len(ref_atoms):.1f})")
    print(
        f"|F| rms at the reference configuration: "
        f"{np.sqrt((f_ref**2).sum(1)).mean():.4f} eV/A\n"
    )

    deltas = [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    direction = rng.normal(size=ref_pos.shape)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)

    print(
        f"{'displacement':>13} | {'exact |F|rms':>12} | "
        f"{'0th err':>11} | {'1st err':>11} | {'ratio':>8}"
    )
    print("-" * 68)

    rows = []
    skipped = []
    rms = lambda x: float(np.sqrt((x**2).sum(1).mean()))  # noqa: E731
    for d in deltas:
        at = ref_atoms.copy()
        at.set_positions(ref_pos + d * direction)
        try:
            sub.mode = "exact"
            f_ex = forces(calc, at)
            sub.mode = "zeroth"
            f_0 = forces(calc, at)
            sub.mode = "first"
            f_1 = forces(calc, at)
        except NeighbourListChangedError as exc:
            skipped.append((d, str(exc)))
            print(
                f"{d:13.1e} |   neighbour-list rebuild ({exc}); "
                f"not comparable to the reference"
            )
            continue

        e0, e1 = rms(f_0 - f_ex), rms(f_1 - f_ex)
        rows.append((d, rms(f_ex), e0, e1))
        ratio = e0 / e1 if e1 > 0 else float("inf")
        print(f"{d:13.1e} | {rms(f_ex):12.5f} | {e0:11.5f} | {e1:11.3e} | {ratio:8.3g}")

    sub.close()
    if skipped:
        print(
            f"\n{len(skipped)} point(s) skipped on a neighbour-list rebuild. "
            f"In Frost this is a generation-counter event, not an error."
        )

    d = np.array([r[0] for r in rows])
    e0 = np.array([r[2] for r in rows])
    e1 = np.array([r[3] for r in rows])

    print("\n=== verdicts ===")

    # Theorem 6.1: zeroth-order error is flat in displacement, and nonzero at 0.
    spread = e0.max() / max(e0.min(), 1e-30)
    print(f"zeroth-order error at zero displacement : {e0[0]:.5f} eV/A")
    print(f"zeroth-order error spread over 3 decades: {spread:.2f}x")
    flat = spread < 3.0 and e0[0] > 1e-3
    print(
        f"  {'CONFIRMED' if flat else 'NOT CONFIRMED'} Theorem 6.1: the error "
        f"is flat in displacement and does not vanish as it goes to zero"
    )

    # Theorem 6.2: first-order error vanishes at zero and grows with displacement.
    nz = d > 0
    if nz.sum() >= 2 and (e1[nz] > 0).all():
        slope = np.polyfit(np.log10(d[nz]), np.log10(e1[nz]), 1)[0]
    else:
        slope = float("nan")
    print(f"\nfirst-order error at zero displacement  : {e1[0]:.3e} eV/A")
    print(
        f"first-order error scaling exponent      : {slope:.2f}  (theory: 1 or better)"
    )
    ok1 = e1[0] < 1e-8 and slope > 0.8
    print(
        f"  {'CONFIRMED' if ok1 else 'NOT CONFIRMED'} Theorem 6.2: the error "
        f"is a genuine perturbation that vanishes with displacement"
    )

    if nz.sum():
        k = int(np.argmax(nz))
        print(
            f"\nAt the tightest nonzero displacement ({d[k]:.0e} A) the "
            f"zeroth-order cache is {e0[k] / e1[k]:,.0f}x worse than the "
            f"first-order one."
        )
        print(
            f"Zeroth-order error as a fraction of the force itself: "
            f"{e0[0] / rows[0][1]:.1%} of |F|rms, at zero displacement."
        )
    return 0 if (flat and ok1) else 1


if __name__ == "__main__":
    raise SystemExit(main())
