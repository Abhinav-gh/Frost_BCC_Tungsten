"""How sparse is a learned closure's refresh demand in MHD turbulence? (new_directions.md §13)

Spatial multirate pays only where the need to refresh is concentrated: a block
whose closure inputs barely changed can keep its frozen closure output. In MD
with MACE-class teachers it does not pay, for two measured reasons:
- homogeneous solids change everywhere at once (E1: dirty fraction 0.24-0.89);
- the receptive-field halo makes a block refresh cost (L_b + 2r)^3 / L_b^3 per
  refreshed atom.

On a grid the halo is a stencil of a few cells, so the question is only the first
one: is turbulence intermittent enough that most blocks are quiet?

For each regime of The Well's MHD_64 (compressible MHD turbulence, 64^3, 100
snapshots 0.01 time units apart; the dataset does not say how many solver steps
that is), this computes the change between consecutive
snapshots of the inputs a subgrid closure reads:
- the velocity-gradient tensor, which drives the Smagorinsky / gradient-model
  stress;
- the magnetic-field-gradient tensor, which drives the EMF closure.

Both are aggregated over 8^3-cell blocks (512 blocks). Reported per regime:
- the fraction of blocks carrying 50% and 80% of the total change (concentration);
- the dirty fraction phi(tau): blocks whose change exceeds tau times the
  domain-mean change;
- how both vary with the sonic and Alfvenic Mach numbers. These are the flow
  counterpart of "molecule and temperature".

CPU and numpy only; no model is run.

Output: data/results/e9/mhd_refresh_sparsity.json

Usage
  LANE=check scripts/safe_run.sh e9_mhd_sparsity .venvs/frost-mhd/bin/python \\
      experiments/e9_direction/mhd_refresh_sparsity.py
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "datasets" / "the_well" / "MHD_64" / "data"
OUT = ROOT / "data" / "results" / "e9" / "mhd_refresh_sparsity.json"
BLOCK = 8
TAUS = (0.25, 0.5, 1.0, 2.0)


def gradient_tensor(field: np.ndarray) -> np.ndarray:
    """Periodic central differences: (64,64,64,3) -> (64,64,64,3,3), in grid units."""
    grads = []
    for axis in range(3):
        grads.append(0.5 * (np.roll(field, -1, axis=axis) - np.roll(field, 1, axis=axis)))
    return np.stack(grads, axis=-1)


def block_norm(x: np.ndarray) -> np.ndarray:
    """Frobenius norm aggregated per 8^3 block: (64,64,64,...) -> (8,8,8)."""
    sq = (x.reshape(x.shape[:3] + (-1,)) ** 2).sum(-1)
    n = sq.shape[0] // BLOCK
    return np.sqrt(sq.reshape(n, BLOCK, n, BLOCK, n, BLOCK).sum((1, 3, 5)))


def concentration(change: np.ndarray, share: float) -> float:
    """Smallest fraction of blocks that carries ``share`` of the total change."""
    v = np.sort(change.ravel())[::-1]
    cum = np.cumsum(v) / v.sum()
    return float((np.searchsorted(cum, share) + 1) / v.size)


def main() -> int:
    files = sorted(DATA.glob("*/*.hdf5"))
    seen, rows = set(), []
    for path in files:
        with h5py.File(path, "r") as fh:
            ma, ms = float(fh["scalars/Ma"][()]), float(fh["scalars/Ms"][()])
            if (ma, ms) in seen:
                continue
            seen.add((ma, ms))
            v = fh["t1_fields/velocity"]
            b = fh["t1_fields/magnetic_field"]
            stats = {"u": [], "b": []}
            for t in range(20, 99, 4):  # skip the start-up transient; subsample in time
                for name, ds in (("u", v), ("b", b)):
                    g0 = gradient_tensor(np.asarray(ds[0, t], dtype=np.float64))
                    g1 = gradient_tensor(np.asarray(ds[0, t + 1], dtype=np.float64))
                    change = block_norm(g1 - g0)
                    level = block_norm(g0)
                    stats[name].append({
                        "rel_change_domain": float(change.sum() / level.sum()),
                        "top50": concentration(change, 0.5),
                        "top80": concentration(change, 0.8),
                        "phi": {str(tau): float((change > tau * change.mean()).mean())
                                for tau in TAUS},
                    })
        row = {"Ma": ma, "Ms": ms, "file": str(path.relative_to(DATA))}
        for name in ("u", "b"):
            s = stats[name]
            row[name] = {
                "rel_change_per_snapshot": float(np.mean([x["rel_change_domain"] for x in s])),
                "blocks_carrying_50pct": float(np.mean([x["top50"] for x in s])),
                "blocks_carrying_80pct": float(np.mean([x["top80"] for x in s])),
                "phi": {tau: float(np.mean([x["phi"][tau] for x in s])) for tau in s[0]["phi"]},
            }
        rows.append(row)
        print(f"Ma {ma:4.1f} Ms {ms:4.1f} | grad-u change/snapshot {row['u']['rel_change_per_snapshot']:.3f}"
              f" | 50% of change in {row['u']['blocks_carrying_50pct']:.2f} of blocks, 80% in "
              f"{row['u']['blocks_carrying_80pct']:.2f} | phi(1x mean) {row['u']['phi']['1.0']:.2f}"
              f" phi(2x) {row['u']['phi']['2.0']:.2f} || grad-B: 50% in "
              f"{row['b']['blocks_carrying_50pct']:.2f}, 80% in {row['b']['blocks_carrying_80pct']:.2f},"
              f" phi(2x) {row['b']['phi']['2.0']:.2f}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
