"""The invalidation predicate and its sensitivity surrogate.

An edge is dirty when the force error its staleness induces exceeds the
per-edge budget:

    dirty(i, j)  <=>  S_hat(r_ij, Z_i, Z_j) * ||dr_ij||  >  eps_edge

Two facts from `Frost.pdf` shape this, and both are easy to get wrong.

**The sensitivity is a curvature, not a slope.** Molecular dynamics consumes
gradients. With a first-order cache the residual force error is governed by the
second derivative of the edge feature map (Corollary 6.3), so the surrogate
estimates ``||d2 g / d r^2||``, not ``||d g / d r||``. Thresholding on the slope
would be the zeroth-order design, which Theorem 6.1 shows does not approximate
the force at all -- it deletes it.

**The surrogate must be an upper envelope, not a fit.** Under-estimating
``S_hat`` lets genuinely stale edges pass the test and silently violates the
error budget. Over-estimating merely costs efficiency. Every step of the
calibration in Appendix B therefore pushes upward: a maximum rather than a
mean, monotone envelope smoothing, and a safety factor.

Budget apportionment across the ``n_bar`` edges incident to an atom uses
``eps / n_bar**gamma`` with gamma in [0.5, 1]:

    gamma = 0.5   independent errors, optimistic, fastest
    gamma = 1.0   fully coherent errors, provably safe
    measured      calibrated from a pilot run (experiment E6)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Budget apportionment modes, Frost.pdf section 4.6.3.
GAMMA = {"independent": 0.5, "worst_case": 1.0}


@dataclass
class SensitivityTable:
    """Upper-envelope surrogate ``S_hat(r, Z_i, Z_j)``, binned by distance.

    Indexed by species pair and distance bin so the runtime predicate is a
    table lookup rather than an autodiff call. For ``n_z`` species and
    ``n_bins`` bins the table is ``n_z**2 * n_bins`` floats -- a few kB, small
    enough to live in shared memory beside the fused predicate kernel.
    """

    species: np.ndarray  # (n_z,) atomic numbers, sorted
    cutoff: float
    table: np.ndarray  # (n_z, n_z, n_bins) float64, an upper bound
    safety: float = 1.2

    @property
    def n_bins(self) -> int:
        return int(self.table.shape[-1])

    def _index(self, z: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.species, z)

    def bin_of(self, r: np.ndarray) -> np.ndarray:
        b = (np.asarray(r) / self.cutoff * self.n_bins).astype(np.int64)
        return np.clip(b, 0, self.n_bins - 1)

    def __call__(self, r: np.ndarray, z_i: np.ndarray, z_j: np.ndarray) -> np.ndarray:
        """Sensitivity for each edge, in eV/A per unit of displacement."""
        return self.table[self._index(z_i), self._index(z_j), self.bin_of(r)]

    def epsilon_edge(
        self, eps: float, mean_degree: float, mode: str | float = "independent"
    ) -> float:
        """Apportion a total force tolerance across one atom's edges."""
        gamma = GAMMA[mode] if isinstance(mode, str) else float(mode)
        return float(eps) / float(mean_degree) ** gamma

    # -- persistence -------------------------------------------------------
    def save(self, path) -> None:
        np.savez_compressed(
            path,
            species=self.species,
            cutoff=self.cutoff,
            table=self.table,
            safety=self.safety,
        )

    @classmethod
    def load(cls, path) -> SensitivityTable:
        d = np.load(path)
        return cls(
            species=d["species"],
            cutoff=float(d["cutoff"]),
            table=d["table"],
            safety=float(d["safety"]),
        )

    @classmethod
    def constant(
        cls, species, cutoff: float, value: float, n_bins: int = 64
    ) -> SensitivityTable:
        """A flat table. Useful as a control: it is what a fixed geometric
        threshold amounts to, and the design rejects it precisely because
        sensitivity varies by an order of magnitude across species pairs and
        bond lengths (Frost.pdf appendix K)."""
        sp = np.sort(np.asarray(species, dtype=np.int64))
        return cls(sp, cutoff, np.full((len(sp), len(sp), n_bins), float(value)))


def upper_envelope(samples: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Fill unmeasured distance bins conservatively, without flattening the rest.

    Sensitivity varies by an order of magnitude across species pairs and bond
    lengths, and that variation is the entire reason the table is indexed by
    distance rather than being one number. An earlier version propagated a
    running maximum both forwards and backwards across bins, which filled every
    cell with the global maximum and destroyed exactly the structure the table
    exists to hold.

    So: a bin with data keeps its measured value. A bin without data inherits
    the maximum of the nearest measured bins on either side, which is an upper
    bound there and leaves measured bins untouched. A row with no data at all
    falls back to the global maximum, which is safe and rare.
    """
    out = np.array(samples, dtype=np.float64, copy=True)
    flat = out.reshape(-1, out.shape[-1])
    seen = counts.reshape(-1, counts.shape[-1]) > 0
    global_max = float(flat[seen].max()) if seen.any() else 0.0

    n_bins = flat.shape[1]
    for row in range(flat.shape[0]):
        have = np.flatnonzero(seen[row])
        if have.size == 0:
            flat[row, :] = global_max
            continue
        for b in range(n_bins):
            if seen[row, b]:
                continue
            # nearest measured bin below and above; bound by the larger
            lo = have[have < b]
            hi = have[have > b]
            candidates = []
            if lo.size:
                candidates.append(flat[row, lo[-1]])
            if hi.size:
                candidates.append(flat[row, hi[0]])
            flat[row, b] = max(candidates)
    return flat.reshape(out.shape)


def is_dirty(
    displacement: np.ndarray,
    sensitivity: np.ndarray,
    eps_edge: float,
    age: np.ndarray | None = None,
    tau_max: int | None = None,
) -> np.ndarray:
    """The predicate of Equation 14, with the age trigger of section 4.6.1.

    The age trigger bounds worst-case staleness for an edge whose geometry
    oscillates without ever crossing the threshold, which would otherwise let
    slow systematic drift accumulate beneath the predicate's resolution.
    """
    dirty = sensitivity * displacement > eps_edge
    if age is not None and tau_max is not None:
        dirty = dirty | (age >= tau_max)
    return dirty
