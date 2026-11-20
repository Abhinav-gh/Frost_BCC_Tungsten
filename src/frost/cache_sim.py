"""Offline per-edge cache replay: the mechanism of E1.

No cache is implemented here in the runtime sense. A logged trajectory is
replayed and a cache is *simulated*, so that the dirty fraction phi(eps) can be
measured before any CUDA is written. This is the whole point of Gate 1: the
decisive measurement needs a logger and a replay, not a system.

What the simulator has to get right, and what a naive version gets wrong:

* **Per-edge reference times.** Each edge carries its own ``t0``. A global
  reference time would reduce Frost to a fixed-stride scheme, which is the
  baseline it has to beat, not the thing being measured.
* **Unwrapped displacement.** ``dr`` is taken against the logged unwrapped
  positions. See `frost.edgekey`.
* **Neighbour-list rebuilds.** When the list is rebuilt, edges appear and
  disappear. A new edge is dirty by definition; a departing edge is
  tombstoned. Matching entries across a rebuild is done on the
  ``(i, j, image)`` key, never on a raw index.
* **Both cache orders from one pass.** The zeroth- and first-order dirty
  fractions come from the same logs, differing only in whether the threshold
  is on slope or curvature. That is what gives Gate 1 two independent chances
  to pass at no extra experimental cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from frost.edgekey import edge_keys
from frost.predicate import SensitivityTable


@dataclass
class ReplayResult:
    """Per-step outcome of one (eps, order) replay."""

    eps: float
    order: int
    system: str
    temperature: float
    dirty_fraction: np.ndarray  # (n_steps,)
    n_edges: np.ndarray  # (n_steps,)
    refresh_interval: np.ndarray  # per-edge intervals, pooled
    rebuild_steps: np.ndarray
    mean_degree: float
    eps_edge: float
    gamma: float

    @property
    def phi(self) -> float:
        """Steady-state dirty fraction: the mean after the warm-up transient.

        The first step is excluded because a cold cache is dirty by definition
        and would bias phi upward on a short run. E28 measures the warm-up cost
        separately rather than hiding it here.
        """
        d = self.dirty_fraction
        return float(np.mean(d[1:])) if len(d) > 1 else float(d[0])

    @property
    def cv_tau(self) -> float:
        """Coefficient of variation of the per-edge refresh interval.

        This is the pre-registered statistic that explains when spatial
        adaptivity pays: a fixed stride is optimal exactly when CV is zero
        (Frost.pdf section 20.3). Reporting it turns a possible negative result
        into a characterisation.
        """
        t = self.refresh_interval
        t = t[t > 0]
        if t.size < 2:
            return 0.0
        return float(np.std(t) / np.mean(t))

    def summary(self) -> dict:
        return {
            "system": self.system,
            "temperature": self.temperature,
            "eps_meV_per_A": self.eps,
            "order": self.order,
            "eps_edge_meV_per_A": self.eps_edge,
            "gamma": self.gamma,
            "phi": self.phi,
            "phi_p50": float(np.median(self.dirty_fraction[1:])),
            "phi_p99": float(np.percentile(self.dirty_fraction[1:], 99)),
            "cv_tau": self.cv_tau,
            "mean_refresh_interval": float(np.mean(self.refresh_interval))
            if self.refresh_interval.size
            else 0.0,
            "mean_degree": self.mean_degree,
            "n_steps": len(self.dirty_fraction),
            "n_rebuilds": len(self.rebuild_steps),
        }


def replay(
    reader,
    sensitivity: SensitivityTable,
    *,
    eps_meV_per_A: float,
    order: int = 1,
    gamma: str | float = "independent",
    tau_max: int | None = None,
    max_steps: int | None = None,
    stride: int = 1,
    model_cutoff: float | None = None,
    geometric_threshold: float | None = None,
) -> ReplayResult:
    """Replay one trajectory under a simulated per-edge cache.

    Pass ``geometric_threshold`` to threshold on displacement in Angstrom
    directly, bypassing the sensitivity surrogate. That is what the Gate 1
    measurement wants: the surrogate converts a force tolerance into a
    geometric one, but the gate asks the question the other way round -- what
    dirty fraction does a given geometric threshold produce, and what force
    error does that threshold actually cost. Measuring both directly and
    crossing them avoids making the gate depend on the quality of a fit.

    The whole computation is vectorised **within a neighbour-list generation**,
    which is the observation that makes E1 tractable at all. Between rebuilds
    the edge list is fixed, so the per-edge key array is identical at every
    step and the cache state can live in plain arrays indexed by row. Only at a
    rebuild is any matching needed, and that is one `np.intersect1d` on the
    keys rather than a dictionary lookup per edge per step.

    The difference is not a constant factor. A per-edge Python loop over this
    trajectory set is 134 billion iterations across the tolerance sweep; the
    vectorised form is a few thousand numpy calls.

    Parameters
    ----------
    reader
        An open `frost.trajectory.TrajectoryReader`.
    sensitivity
        Upper-envelope surrogate. For ``order=0`` it should hold slope values,
        for ``order=1`` curvature values (Corollary 6.3).
    eps_meV_per_A
        Total force tolerance. Apportioned across an atom's edges by ``gamma``.
    order
        0 reproduces the inadmissible zeroth-order cache, kept because the
        contrast is the measurement. 1 is the admissible design.
    """
    numbers = reader.numbers
    cut = model_cutoff if model_cutoff is not None else float(reader.attrs["cutoff"])
    n_steps = reader.n_steps if max_steps is None else min(reader.n_steps, max_steps)
    eps_eV = eps_meV_per_A * 1e-3  # sensitivities are fitted in eV per Angstrom

    dirty_frac: list[float] = []
    n_edges_per_step: list[int] = []
    rebuilds: list[int] = []
    intervals: list[np.ndarray] = []

    last_gen = -1
    keys = sel = zi = zj = None
    r_ref = t_ref = None
    eps_edge = None
    mean_degree = 0.0

    for step in range(0, n_steps, stride):
        gen = reader.generation_at(step)

        if gen != last_gen:
            # New edge list. Select the edges inside the model cutoff once for
            # this generation, then carry forward any cache entry whose key
            # survives. Entries for departed edges are simply dropped.
            i_all, j_all, S_all, _ = reader.neighbour_list_at(step)
            vec_all = reader.edge_vectors_at(step)[0]
            sel = np.linalg.norm(vec_all, axis=1) <= cut
            i, j, S = i_all[sel], j_all[sel], S_all[sel]
            new_keys = edge_keys(i, j, S)
            zi, zj = numbers[i], numbers[j]

            new_r = np.full((len(new_keys), 3), np.nan)
            new_t = np.full(len(new_keys), -1, dtype=np.int64)
            if keys is not None:
                _, old_idx, new_idx = np.intersect1d(
                    keys, new_keys, return_indices=True
                )
                new_r[new_idx] = r_ref[old_idx]
                new_t[new_idx] = t_ref[old_idx]
            keys, r_ref, t_ref = new_keys, new_r, new_t
            rebuilds.append(step)
            last_gen = gen

            if mean_degree == 0.0:
                mean_degree = len(keys) / reader.n_atoms
            if eps_edge is None:
                eps_edge = (
                    geometric_threshold
                    if geometric_threshold is not None
                    else sensitivity.epsilon_edge(eps_eV, mean_degree, gamma)
                )

        pos = np.asarray(reader.positions[step])
        vec = reader.edge_vectors_from(pos, step)[sel]
        d = np.linalg.norm(vec, axis=1)
        s_hat = (
            np.ones_like(d)
            if geometric_threshold is not None
            else sensitivity(d, zi, zj)
        )

        # A never-computed edge (NaN reference) is dirty by definition.
        fresh = np.isnan(r_ref[:, 0])
        delta = np.where(fresh[:, None], 0.0, vec - np.nan_to_num(r_ref))
        dr = np.linalg.norm(delta, axis=1)

        dirty = fresh | (s_hat * dr > eps_edge)
        if tau_max is not None:
            dirty |= (step - t_ref) >= tau_max

        hit = np.flatnonzero(dirty)
        if hit.size:
            prev = t_ref[hit]
            seen = prev >= 0
            if seen.any():
                intervals.append((step - prev[seen]).astype(np.float64))
            r_ref[hit] = vec[hit]
            t_ref[hit] = step

        dirty_frac.append(float(dirty.mean()))
        n_edges_per_step.append(len(keys))

    refresh_interval = (
        np.concatenate(intervals) if intervals else np.empty(0, dtype=np.float64)
    )
    return ReplayResult(
        eps=eps_meV_per_A,
        order=order,
        system=str(reader.attrs.get("system", "?")),
        temperature=float(reader.attrs.get("temperature_K", 0)),
        dirty_fraction=np.asarray(dirty_frac),
        n_edges=np.asarray(n_edges_per_step),
        refresh_interval=refresh_interval,
        rebuild_steps=np.asarray(rebuilds),
        mean_degree=mean_degree,
        eps_edge=float(eps_edge) if eps_edge else 0.0,
        gamma=GAMMA_VALUE(gamma),
    )


def GAMMA_VALUE(gamma) -> float:  # noqa: N802
    from frost.predicate import GAMMA

    return GAMMA[gamma] if isinstance(gamma, str) else float(gamma)


def fixed_stride(
    reader, *, k: int, max_steps: int | None = None, model_cutoff: float | None = None
) -> ReplayResult:
    """The baseline that decides the paper: recompute everything every k steps.

    By Proposition 20.5 this is a point in Frost's own configuration space
    (eps = infinity, tau_max = k, staggering off), so Frost cannot lose to it.
    The measurement is how large the gap is, and CV(tau) predicts that in
    advance.
    """
    cut = model_cutoff if model_cutoff is not None else float(reader.attrs["cutoff"])
    n_steps = reader.n_steps if max_steps is None else min(reader.n_steps, max_steps)
    frac, n_edges = [], []
    for step in range(n_steps):
        vec, _i, _j, _S, _ = reader.edge_vectors_at(step)
        inside = np.linalg.norm(vec, axis=1) <= cut
        n = int(inside.sum())
        frac.append(1.0 if step % k == 0 else 0.0)
        n_edges.append(n)
    return ReplayResult(
        eps=float("inf"),
        order=1,
        system=str(reader.attrs.get("system", "?")),
        temperature=float(reader.attrs.get("temperature_K", 0)),
        dirty_fraction=np.asarray(frac),
        n_edges=np.asarray(n_edges),
        refresh_interval=np.full(1, float(k)),
        rebuild_steps=np.empty(0),
        mean_degree=float(np.mean(n_edges)) / reader.n_atoms,
        eps_edge=float("inf"),
        gamma=0.0,
    )
