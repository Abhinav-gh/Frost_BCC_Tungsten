"""A liquid-Cu teacher trajectory for the liquids track (new_directions.md R10, test 9).

Frost's only liquid so far, water, fails the dynamics contract (plan section 1.13):
its hydrogens carry a broadband 4% student error at 9 fs stretch periods. Liquid Cu
is the simplest liquid Frost can test: no hydrogen, the same 500-atom cell and
pipeline as solid Cu, and diffusion as the observable a liquid adds.

Protocol, with MACE-MP-0 medium through ``frost.trajectory.run_md`` (NVT Langevin,
unwrapped positions; the same writer as E1):
  1. melt: 5 ps at 2500 K (well above Cu's 1358 K, so a superheated crystal cannot
     survive);
  2. equilibrate: 10 ps at 1600 K;
  3. log: 10 ps at 1600 K, 1 fs, every step -> data/trajectories/cu500_T1600.h5.

It then checks that the cell is liquid: the mean-squared displacement over the
logged 10 ps must grow linearly (D > 0.05 A^2/ps). A crystal gives a plateau.

Usage (guarded launcher; GPU)
  scripts/safe_run.sh liquid_cu_traj .venvs/frost-mace/bin/python \\
      experiments/e9_direction/liquid_cu_trajectory.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "trajectories" / "cu500_T1600.h5"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--melt-steps", type=int, default=5000)
    ap.add_argument("--equil-steps", type=int, default=10000)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if OUT.exists() and not args.overwrite:
        raise SystemExit(f"{OUT} exists; pass --overwrite")

    from mace.calculators import mace_mp

    from frost.systems import cu500
    from frost.trajectory import run_md

    atoms = cu500()
    calc = mace_mp(model="medium", device=args.device, default_dtype="float32")
    # run_md works on a copy and, with out=None, runs only its equilibration and returns
    # the atoms: the melt must go in equilibrate_steps and its result must be carried on
    melted = run_md(atoms, calc, steps=0, timestep_fs=1.0, temperature_K=2500.0,
                    out=None, cutoff=6.0, equilibrate_steps=args.melt_steps)
    run_md(melted, calc, steps=args.steps, timestep_fs=1.0, temperature_K=1600.0, out=OUT,
           cutoff=6.0, equilibrate_steps=args.equil_steps)

    with h5py.File(OUT, "r") as fh:
        x = np.asarray(fh["positions"][:], dtype=np.float64)
        temps = np.asarray(fh["temperature"][:])
    lags = np.arange(100, 5001, 100)
    msd = np.array([np.mean(np.sum((x[lag:] - x[:-lag]) ** 2, axis=-1)) for lag in lags])
    slope = float(np.polyfit(lags * 1e-3, msd, 1)[0])  # A^2 per ps
    d = slope / 6.0
    report = {"path": str(OUT.relative_to(ROOT)), "mean_T_K": float(temps.mean()),
              "diffusion_A2_per_ps": d, "msd_at_1ps": float(msd[9]), "msd_at_5ps": float(msd[-1]),
              "liquid": bool(d > 0.05)}
    print(json.dumps(report, indent=1))
    (ROOT / "data" / "results" / "e9" / "liquid_cu_trajectory.json").write_text(
        json.dumps(report, indent=1))
    if not report["liquid"]:
        raise SystemExit("the cell did not melt: D is below 0.05 A^2/ps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
