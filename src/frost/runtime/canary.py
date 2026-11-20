"""Canary certification of a faster schedule, without a reference simulation (gap N5).

A CFD or MHD closure's hold is judged by long-run statistics, and the reference that scores them (a
DNS) does not exist at run time. Ensembles do: an LES is run as many trajectories. So the runtime
certifies a candidate schedule against a *trusted* one inside the same run, as a canary release is
judged against production:

- the trusted group of members runs the schedule that was validated (once, offline) against a
  reference; the canary group runs the faster candidate;
- after a burn-in, each member's statistics are averaged over time; for every statistic the two
  groups' member means are compared with Welch's t-test;
- the candidate is promoted (``verdict == "promote"``) when no statistic differs by more than the
  tolerance at the given confidence, and refused when one does; until then it is undecided.

Both groups advance through their own ``HoldBackend`` (``frost.runtime.hold``) with the executor's
clauses, so a canary that goes non-finite or breaks a hold-error bound is rolled back as usual.
``stats_fn(u_hat) -> (members, n_stats)`` defines what is compared.
"""

from __future__ import annotations

import math

import numpy as np


def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Difference of means b - a and its standard error (Welch), over members (axis 0)."""
    ma, mb = a.mean(0), b.mean(0)
    va = a.var(0, ddof=1) / a.shape[0] if a.shape[0] > 1 else np.zeros_like(ma)
    vb = b.var(0, ddof=1) / b.shape[0] if b.shape[0] > 1 else np.zeros_like(mb)
    return mb - ma, np.sqrt(va + vb)


class Canary:
    def __init__(self, trusted, candidate, stats_fn, names, burn_in: float, tolerance: dict,
                 z: float = 2.0):
        """``trusted``/``candidate``: (backend, executor) pairs; ``tolerance``: stat name -> largest
        admissible relative difference of the group means."""
        self.trusted, self.candidate = trusted, candidate
        self.stats_fn, self.names = stats_fn, list(names)
        self.burn_in, self.tolerance, self.z = burn_in, tolerance, z
        self.sums = {"trusted": None, "candidate": None}
        self.counts = {"trusted": 0, "candidate": 0}

    def _accumulate(self, group: str, u_hat) -> None:
        s = np.asarray(self.stats_fn(u_hat), dtype=np.float64)  # (members, n_stats)
        self.sums[group] = s if self.sums[group] is None else self.sums[group] + s
        self.counts[group] += 1

    def advance(self, until: float, record_every: int) -> None:
        """Advance both groups to ``until`` steps, recording statistics about every ``record_every`` steps.

        Each group records at multiples of its own interval (rounded up from ``record_every``): asking
        the executor to stop between a hold's boundaries would cut the hold short (the first version of
        this experiment ran holds of 16 and 32 as holds of 10)."""
        for group, (backend, executor) in (("trusted", self.trusted), ("candidate", self.candidate)):
            k = max(1, getattr(executor, "fixed_k", 0) or 1)
            stride = k * max(1, -(-record_every // k))
            while backend.time() + stride <= until:
                executor.run(backend.time() + stride)
                if backend.time() >= self.burn_in:
                    self._accumulate(group, backend.u)

    def verdict(self) -> dict:
        if not self.counts["trusted"] or not self.counts["candidate"]:
            return {"verdict": "undecided", "why": "burn-in"}
        a = self.sums["trusted"] / self.counts["trusted"]
        b = self.sums["candidate"] / self.counts["candidate"]
        diff, se = welch(a, b)
        scale = np.abs(a.mean(0))
        out, refuse, undecided = {}, [], []
        for i, name in enumerate(self.names):
            if name not in self.tolerance:
                continue
            rel, rel_se = float(diff[i] / max(scale[i], 1e-30)), float(se[i] / max(scale[i], 1e-30))
            tol = self.tolerance[name]
            out[name] = {"rel_diff": rel, "rel_se": rel_se, "tol": tol}
            if abs(rel) > tol and abs(rel) > self.z * rel_se:
                refuse.append(name)  # a significant difference whose estimate exceeds the tolerance
            elif abs(rel) + self.z * rel_se > tol:
                undecided.append(name)  # equivalence within the tolerance not yet shown (TOST)
        v = "refuse" if refuse else "undecided" if undecided else "promote"
        return {"verdict": v, "refused_on": refuse, "undecided_on": undecided, "stats": out,
                "records": dict(self.counts)}


def members_needed(rel_sd: float, tolerance: float, z: float = 2.0) -> int:
    """Members per group for a promote decision when the true difference is zero."""
    return max(2, math.ceil(2.0 * (z * rel_sd / tolerance) ** 2))
