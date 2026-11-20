"""Si-O-Si angle distribution from stored teacher frames, to gate a sampling time step.

The gate on a larger step currently uses the mean potential energy and the radial
distribution, both of which are configurational *averages*. BAOAB's bias is
O(dt^2) and can move the distribution somewhere an average does not look: a
radial distribution integrates over orientation, so a bond-bending distortion
survives it. The Si-O-Si angle is sensitive to exactly the modes a large step
distorts first, which is why it is the second observable.

Reads the frames sample.py stores with --save-teacher-ref and reports, per arm,
the angle histogram, its mean and its shift against the reference arm. The first
argument is the reference; the rest are judged against it.

It settled the teacher's time step for the statistics contract (plan section
1.15): at 4 fs the two seeds bracket the 1 fs reference (-0.060 and +0.032
degrees against 147.430) and differ from each other by more than either differs
from it, so there is no angular distortion to find, and the 4 fs baseline is
admissible. Read shifts against the seed-to-seed spread, never against zero.

Usage
  .venvs/frost-mace/bin/python experiments/e8_statistics/angle_gate.py \
      s87_ang_dt1 s87_ang_dt2 s87_ang_dt4 s87_ang_dt4_s1
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

SI, OX = 14, 8
R_SIO = 2.0  # A, first minimum of the Si-O radial distribution


def minimum_image(d: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Displacements folded into the nearest image, for an orthorhombic cell."""
    lengths = np.diag(cell)
    return d - lengths * np.round(d / lengths)


def angles(frame: np.ndarray, numbers: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Every Si-O-Si angle in one frame, in degrees.

    An oxygen bridges two silicons; the angle at that oxygen is the bond-bending
    coordinate. Pairs are found per oxygen from the Si neighbours within R_SIO.
    """
    si = np.flatnonzero(numbers == SI)
    ox = np.flatnonzero(numbers == OX)
    out = []
    si_pos = frame[si]
    for o in ox:
        d = minimum_image(si_pos - frame[o], cell)
        r = np.linalg.norm(d, axis=1)
        near = np.flatnonzero(r < R_SIO)
        for a in range(len(near)):
            for b in range(a + 1, len(near)):
                u, v = d[near[a]], d[near[b]]
                cos = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
                out.append(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    return np.asarray(out)


def main() -> int:
    import h5py

    with h5py.File(ROOT / "data/trajectories/silica1152_T300.h5", "r") as fh:
        numbers = np.asarray(fh.attrs["numbers"], dtype=np.int64)
        cell = np.asarray(fh.attrs["cell"], dtype=np.float64)

    edges = np.linspace(120.0, 180.0, 61)
    reference = None
    print(f"{'arm':22s} {'frames':>7s} {'mean angle':>11s} {'shift':>8s} {'hist L1':>8s} {'sd':>7s}")
    for tag in sys.argv[1:]:
        path = ROOT / "data/results/e8" / (tag if tag.endswith(".npz") else f"{tag}.npz")
        if not path.exists():
            print(f"{tag:22s} missing: {path.name}")
            continue
        frames = np.load(path)["frames"]
        every = max(1, len(frames) // 120)  # 120 frames is plenty for an angle histogram
        collected = np.concatenate([angles(np.asarray(f, dtype=np.float64), numbers, cell)
                                    for f in frames[::every]])
        hist, _ = np.histogram(collected, bins=edges, density=True)
        mean, sd = collected.mean(), collected.std()
        if reference is None:
            reference = (hist, mean)
            shift, l1 = "reference", "reference"
        else:
            shift = f"{mean - reference[1]:+.3f}"
            l1 = f"{np.abs(hist - reference[0]).sum() * (edges[1] - edges[0]):.4f}"
        print(f"{tag:22s} {len(frames[::every]):7d} {mean:11.3f} {shift:>8s} {l1:>8s} {sd:7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
