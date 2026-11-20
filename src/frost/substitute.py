"""Substitute a cached surrogate into a model's edge tensor product.

This is how Frost's approximation is measured before Frost exists. A forward
hook replaces one module's output with the value a cache would have produced,
so the induced force error can be read off against exact evaluation on the same
configuration.

The trick that makes the first-order cache exact without materialising a
Jacobian is that the module is trilinear in its three inputs -- node features,
spherical harmonics and radial weights:

    g(a, b, w)  =  TP(a, b, w)                                  exact
    g_0         =  TP(a0, b0, w0)                               zeroth order
    g_1         =  g_0 + TP(da, b0, w0)
                       + TP(a0, db, w0) + TP(a0, b0, dw)        first order

The reference terms are detached constants; the deltas stay attached, and that
is what carries the gradient. Detaching them instead reproduces the failure of
Theorem 6.1 silently: plausible energies, catastrophic forces.

Two facts confirmed with this machinery on MACE-MP-0, both in
`experiments/probes/probe_force_error.py`:

  * zeroth order is wrong by 68% of the force RMS at **zero** displacement,
    and flat in displacement to a factor of 1.00 (Theorem 6.1);
  * first order is exact at zero to float64 machine epsilon and grows with a
    fitted scaling exponent of 1.00 (Theorem 6.2).
"""

from __future__ import annotations

import numpy as np
import torch


class NeighbourListChangedError(RuntimeError):
    """The edge set moved, so the reference is no longer comparable.

    Frost handles this with generation counters; a measurement should refuse
    the comparison rather than silently line up unrelated edges.
    """

    def __init__(self, now: int, ref: int):
        super().__init__(f"edge count {ref} -> {now}")
        self.now, self.ref = now, ref


class EdgeCacheSubstituter:
    """Replace one tensor product's output with a cached surrogate.

    Modes
    -----
    record   capture the reference inputs and output, pass through unchanged
    exact    pass through unchanged
    zeroth   return the detached reference value (inadmissible; Theorem 6.1)
    first    return the trilinear first-order expansion (the real design)
    first_stale_a  first order in edge geometry, neighbour state held stale

    Edge identity
    -------------
    Displacing atoms moves edges across the cutoff, so the reference and the
    displaced configuration generally have different edge lists. Checking only
    that the row COUNT matches is not enough: a reordering that happens to
    preserve the count passes that check and then substitutes one edge's cached
    value into another edge's slot. The result is not an error, it is a
    plausible-looking force that is wrong by more than the force itself --
    measured at 2406 meV/A against a true force RMS of 817.

    Supply `keys` alongside each configuration and the substituter caches only
    the edges present in both, leaving the rest to be computed exactly. That is
    also precisely what Frost does at a neighbour-list rebuild, so the
    measurement matches the semantics it is measuring.
    """

    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.mode = "record"
        self.ref: tuple | None = None
        self.ref_keys: np.ndarray | None = None
        self.cur_keys: np.ndarray | None = None
        self._busy = False
        self._handle = module.register_forward_hook(self._hook)

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

        if self.ref_keys is not None and self.cur_keys is not None:
            # Map cached rows onto current rows by edge identity. Rows with no
            # counterpart keep their exact value, which is what Frost does for
            # an edge that has just entered the neighbour list.
            _, i_ref, i_cur = np.intersect1d(
                self.ref_keys, self.cur_keys, return_indices=True
            )
            if len(i_cur) == 0:
                return None
            idx_cur = torch.as_tensor(i_cur, device=a.device, dtype=torch.long)
            idx_ref = torch.as_tensor(i_ref, device=a.device, dtype=torch.long)
            a0m, b0m, w0m, g0m = (t[idx_ref] for t in (a0, b0, w0, g0))
            am, bm, wm = a[idx_cur], b[idx_cur], w[idx_cur]
        elif a.shape != a0.shape:
            raise NeighbourListChangedError(a.shape[0], a0.shape[0])
        else:
            idx_cur = None
            a0m, b0m, w0m, g0m = a0, b0, w0, g0
            am, bm, wm = a, b, w

        if self.mode == "zeroth":
            sub = g0m  # a constant: dg/dr is identically zero
        elif self.mode == "first":
            self._busy = True
            try:
                sub = g0m + (
                    self.module(am - a0m, b0m, w0m)
                    + self.module(a0m, bm - b0m, w0m)
                    + self.module(a0m, b0m, wm - w0m)
                )
            finally:
                self._busy = False
        elif self.mode == "first_stale_a":
            # First order in this edge's own geometry only, with the incoming
            # node state held at its reference value. This is what a stored
            # geometric Jacobian can deliver at a deeper layer: dg/dr at fixed
            # neighbour features. Both the stale value AND the missing gradient
            # through the node-feature path are what attenuated closure has to
            # account for.
            self._busy = True
            try:
                sub = g0m + (
                    self.module(a0m, bm - b0m, w0m) + self.module(a0m, b0m, wm - w0m)
                )
            finally:
                self._busy = False
        else:
            raise ValueError(self.mode)

        if idx_cur is None:
            return sub
        out = output.clone()
        out[idx_cur] = sub
        return out

    def close(self):
        self._handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def forces(calc, atoms) -> np.ndarray:
    """Force a genuine re-evaluation.

    ASE calculators cache `results` and skip recomputation when the atoms are
    unchanged. Measuring several substitution modes on identical positions
    without clearing that cache returns the first mode's answer every time --
    silently, as an all-zero error.
    """
    calc.reset()
    a = atoms.copy()
    a.calc = calc
    return a.get_forces()


def force_error_at(
    calc,
    sub: EdgeCacheSubstituter,
    reference,
    displaced,
    mode: str = "first",
    ref_keys=None,
    cur_keys=None,
) -> tuple[float, float]:
    """RMS force error of `mode` against exact evaluation on `displaced`.

    Pass `ref_keys` and `cur_keys` to match edges by identity; without them the
    substituter falls back to requiring identical edge counts and refuses the
    comparison otherwise.

    Returns (error, exact force RMS), both in eV/A.
    """
    sub.ref_keys = sub.cur_keys = None
    sub.mode = "record"
    forces(calc, reference)

    sub.ref_keys, sub.cur_keys = ref_keys, cur_keys
    sub.mode = "exact"
    f_exact = forces(calc, displaced)
    sub.mode = mode
    f_cached = forces(calc, displaced)

    err = float(np.sqrt(((f_cached - f_exact) ** 2).sum(1).mean()))
    rms = float(np.sqrt((f_exact**2).sum(1).mean()))
    return err, rms
