"""Fit the sensitivity surrogate S_hat (Frost.pdf appendix B, amended by 6.3).

The predicate needs to convert a force tolerance in meV/A into a geometric
threshold per edge. That conversion is the product of two factors:

    S_e  =  ||dE/dg_e||  x  ||d^k g_e / d r_e^k||
            ^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^
            how much the     how fast the edge
            energy cares     feature map moves
            about this edge

with ``k = 1`` (slope) for the zeroth-order cache and ``k = 2`` (curvature) for
the first-order one. Corollary 6.3 is the reason the admissible design uses
curvature: with the Jacobian stored, the residual error is governed by how the
map bends, not by how steeply it runs.

Both factors come from the same sweep:

* ``||dE/dg_e||`` is one backward pass per frame. Exact, all edges at once.
* ``||J_e||`` and ``||H_e||`` are estimated from how ``g`` responds to small
  random displacements. Across many samples in a distance bin,
  ``||dg|| ~ ||J|| ||dr|| + 0.5 ||H|| ||dr||^2``, so a two-term fit in each bin
  recovers both. This avoids materialising a per-edge Jacobian during
  calibration, which would cost more memory than the cache itself.

**Edges are matched by key, never by row index.** Displacing atoms moves edges
across the cutoff, so the perturbed configuration has a different edge list and
a different ordering. Comparing row *i* of one to row *i* of the other silently
compares unrelated edges. The intersection is taken on ``(i, j, image)``, which
is the same discipline the cache itself uses across a neighbour-list rebuild.

Every reduction is a maximum, never a mean, and the result is passed through a
monotone envelope and inflated by a safety factor. An under-estimate silently
violates the error budget; an over-estimate only costs speed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from frost.edgekey import edge_keys
from frost.predicate import SensitivityTable, upper_envelope


@dataclass
class EdgeMapProbe:
    """Capture a MACE interaction block's edge tensor product output."""

    module: torch.nn.Module
    output: torch.Tensor | None = None

    def __post_init__(self):
        self._h = self.module.register_forward_hook(self._hook)

    def _hook(self, mod, inputs, output):
        self.output = output

    def close(self):
        self._h.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _batch_for(calc, atoms, cutoff: float, device: str):
    """MACE's own batch, plus the edge keys and vectors that identify each row."""
    from mace import data as mdata
    from mace.tools import torch_geometric, utils

    z = utils.AtomicNumberTable([int(x) for x in calc.models[0].atomic_numbers])
    cfg = mdata.config_from_atoms(atoms)
    d = mdata.AtomicData.from_config(cfg, z_table=z, cutoff=cutoff)
    batch = next(iter(torch_geometric.dataloader.DataLoader([d], batch_size=1)))

    src = batch.edge_index[0].numpy().astype(np.int64)
    dst = batch.edge_index[1].numpy().astype(np.int64)
    shifts = batch.shifts.numpy()
    pos = atoms.get_positions()
    vec = pos[dst] - pos[src] + shifts

    cell = np.array(atoms.cell)
    # Recover the integer image from the real-space shift so the key is exact.
    img = np.rint(shifts @ np.linalg.inv(cell)).astype(np.int64)
    keys = edge_keys(src, dst, img)

    # AtomicData builds float32 tensors regardless of the model's dtype, and
    # calling the model directly bypasses the conversion the ASE calculator
    # would have done. Cast explicitly or tensordot raises on mixed dtypes.
    batch = batch.to(device)
    dtype = next(calc.models[0].parameters()).dtype
    for name in ("positions", "shifts", "unit_shifts", "cell", "node_attrs"):
        t = getattr(batch, name, None)
        if torch.is_tensor(t) and t.is_floating_point():
            batch[name] = t.to(dtype)
    return batch, src, dst, vec, keys


def _energy_and_g(calc, batch, probe):
    out = calc.models[0](batch.to_dict(), compute_force=False, training=True)
    return out["energy"].sum(), probe.output


def collect_samples(
    calc,
    frames,
    *,
    site: str,
    cutoff: float,
    displacements=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
    seed: int = 0,
    device: str = "cuda",
) -> dict:
    """Sample (distance, species pair, |dr|, |dg|, |dE/dg|) over frames."""
    module = dict(calc.models[0].named_modules())[site]
    rng = np.random.default_rng(seed)
    cols: dict[str, list] = {k: [] for k in ("r", "zi", "zj", "dr", "dg", "w")}
    n_matched, n_rows = 0, 0

    with EdgeMapProbe(module) as probe:
        for atoms in frames:
            numbers = atoms.get_atomic_numbers()
            b0, src0, dst0, vec0, keys0 = _batch_for(calc, atoms, cutoff, device)

            energy, g0 = _energy_and_g(calc, b0, probe)
            (wgrad,) = torch.autograd.grad(energy, g0, allow_unused=True)
            if wgrad is None:
                raise RuntimeError(
                    f"dE/dg is None at {site}: the tensor product is not on the "
                    "autograd path, so the calibration would be meaningless"
                )
            w_edge = wgrad.detach().norm(dim=1).double().cpu().numpy()
            g0d = g0.detach().double().cpu().numpy()
            d0 = np.linalg.norm(vec0, axis=1)
            pos0 = atoms.get_positions()

            for amp in displacements:
                direction = rng.normal(size=pos0.shape)
                direction /= np.linalg.norm(direction, axis=1, keepdims=True)
                moved = atoms.copy()
                moved.set_positions(pos0 + amp * direction)

                b1, _, _, vec1, keys1 = _batch_for(calc, moved, cutoff, device)
                _, g1 = _energy_and_g(calc, b1, probe)
                g1d = g1.detach().double().cpu().numpy()

                # Align on edge identity, not row order.
                _, ia, ib = np.intersect1d(keys0, keys1, return_indices=True)
                n_matched += len(ia)
                n_rows += len(keys0)
                if len(ia) == 0:
                    continue

                cols["r"].append(d0[ia])
                cols["zi"].append(numbers[src0[ia]])
                cols["zj"].append(numbers[dst0[ia]])
                cols["dr"].append(np.linalg.norm(vec1[ib] - vec0[ia], axis=1))
                cols["dg"].append(np.linalg.norm(g1d[ib] - g0d[ia], axis=1))
                cols["w"].append(w_edge[ia])

    out = {k: (np.concatenate(v) if v else np.empty(0)) for k, v in cols.items()}
    out["matched_fraction"] = n_matched / max(n_rows, 1)
    return out


def fit_tables(
    samples: dict,
    *,
    species: np.ndarray,
    cutoff: float,
    n_bins: int = 64,
    safety: float = 1.2,
    min_count: int = 8,
) -> tuple[SensitivityTable, SensitivityTable]:
    """Fit the zeroth- and first-order sensitivity tables from one sample set.

    Within each (species pair, distance bin) cell we fit

        |dg| ~ a |dr| + b |dr|^2

    in two terms, so ``a`` estimates ``||J||`` and ``2b`` estimates ``||H||``.
    Both are multiplied by the per-edge weight ``||dE/dg||`` and reduced with a
    maximum, because the table must bound rather than describe.
    """
    r, dr, dg, w = samples["r"], samples["dr"], samples["dg"], samples["w"]
    if r.size == 0:
        raise ValueError(
            "no calibration samples: every perturbed configuration failed to "
            "match the reference edge set. Reduce the displacement amplitudes."
        )

    sp = np.sort(np.unique(species).astype(np.int64))
    idx = {int(z): n for n, z in enumerate(sp)}
    shape = (len(sp), len(sp), n_bins)
    s0 = np.zeros(shape)
    s1 = np.zeros(shape)
    counts = np.zeros(shape, dtype=np.int64)
    diag: list[tuple[float, float, float, float]] = []

    bins = np.clip((r / cutoff * n_bins).astype(int), 0, n_bins - 1)
    ii = np.array([idx.get(int(z), 0) for z in samples["zi"].astype(int)])
    jj = np.array([idx.get(int(z), 0) for z in samples["zj"].astype(int)])
    cell_id = (ii * len(sp) + jj) * n_bins + bins

    for cell in np.unique(cell_id):
        sel = np.flatnonzero(cell_id == cell)
        if sel.size < min_count:
            continue
        x, y, wt = dr[sel], dg[sel], w[sel]
        good = (x > 0) & np.isfinite(y) & np.isfinite(x)
        if good.sum() < min_count:
            continue
        x, y, wt = x[good], y[good], wt[good]
        A = np.stack([x, x**2], axis=1)
        try:
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        j_norm = max(float(coef[0]), 0.0)
        h_norm = max(2.0 * float(coef[1]), 0.0)
        resid = y - A @ coef
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else 0.0
        wmax = float(np.max(wt))
        i_, j_, bn = np.unravel_index(int(cell), shape)
        s0[i_, j_, bn] = j_norm * wmax
        s1[i_, j_, bn] = h_norm * wmax
        counts[i_, j_, bn] = int(good.sum())
        diag.append((j_norm, h_norm, r2, float(x.max())))

    if diag:
        d = np.array(diag)
        jn, hn, r2, xmax = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
        ok = (jn > 0) & (hn > 0)
        # 2||J||/||H|| is the displacement at which the quadratic remainder
        # matches the linear term. If it is far below the range actually
        # sampled, the curvature estimate is fitting noise rather than physics.
        cross = 2 * jn[ok] / hn[ok] if ok.any() else np.array([np.nan])
        print(
            f"  fit diagnostics over {len(diag)} cells: "
            f"median R^2 = {np.median(r2):.4f}, "
            f"median 2|J|/|H| = {np.median(cross):.3g} A, "
            f"max |dr| sampled = {xmax.max():.3g} A"
        )
        # Frost.pdf appendix Q estimates H ~ J / l with l about 3 A, from the
        # decay length of the radial envelope. 2|J|/|H| is that same l, measured.
        if 1.0 <= np.median(cross) <= 6.0:
            print(
                "  -> consistent with the design document's estimate of "
                "l = 3 A for the envelope decay length"
            )
        if np.median(cross) > 100 * xmax.max():
            print(
                "  WARNING: the curvature term is unresolved at these "
                "displacements; S_hat(order=1) is not trustworthy"
            )

    s0 = upper_envelope(s0, counts) * safety
    s1 = upper_envelope(s1, counts) * safety
    return (
        SensitivityTable(sp, cutoff, s0, safety),
        SensitivityTable(sp, cutoff, s1, safety),
    )
