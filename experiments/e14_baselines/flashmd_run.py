"""FlashMD (Bigi, Chong, Kristiadi and Ceriotti, NeurIPS 2025) rerun on our GPU through its released package (T1).

FlashMD predicts positions and momenta over long strides with a model trained on trajectories of its own reference
MLIP, and never calls that MLIP during a run. This driver uses only the package's public API, as its README does:
``flashmd.get_pretrained(mlip, stride)`` returns the reference energy model and the stride model,
``flashmd.ase.EnergyCalculator`` wraps the energy model for ASE, and ``flashmd.ase.velocity_verlet.VelocityVerlet`` (or
their ``Langevin``) integrates with the stride model.

Arms, from the same start frame of one of our trajectories (positions and velocities):
- ``reference``: their energy model run by ASE's own velocity Verlet at ``--ref-dt`` (their "traditional MD" check);
- ``flashmd_<s>``: their stride model at s fs, energy rescaled each step (their default);
- ``flashmd_<s>s``: their symplectic stride model (``--symplectic``; pet-omatpes has 2 and 16 fs).

Arms run by separate invocations with the same ``--tag`` merge into one report.

Each arm records velocities at every step of its own integrator and positions every ``--x-fs``, and the wall time of
the timed part (after ``--warm-steps`` steps). Frames go to ``data/results/e14/flashmd_<run>_<tag>_<arm>.npz`` for
``flashmd_score.py`` (frost env), which scores every arm against the reference arm.

Usage (guarded launcher; GPU)
  LANE=gpu1 scripts/safe_run.sh flashmd_cu .venvs/frost-flashmd/bin/python experiments/e14_baselines/flashmd_run.py \\
      --run cu500_T300 --strides 16,32,64 --ps 20 --tag nve20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import ase
import ase.units as units
import h5py
import numpy as np
from ase.md.verlet import VelocityVerlet as AseVerlet
from flashmd import get_pretrained
from flashmd.ase import EnergyCalculator
from flashmd.ase.velocity_verlet import VelocityVerlet as FlashVerlet

ROOT = Path(__file__).resolve().parents[2]
TRAJ = ROOT / "data" / "trajectories"
OUT = ROOT / "data" / "results" / "e14"


def start_atoms(run: str, frame: int) -> ase.Atoms:
    with h5py.File(TRAJ / f"{run}.h5", "r") as fh:
        numbers = np.asarray(fh.attrs["numbers"])
        cell = np.asarray(fh.attrs["cell"], dtype=np.float64)
        x = np.asarray(fh["positions"][frame], dtype=np.float64)
        v = np.asarray(fh["velocities"][frame], dtype=np.float64)  # ASE units, as our trajectories store them
    atoms = ase.Atoms(numbers=numbers, positions=x, cell=cell, pbc=True)
    atoms.set_velocities(v)
    return atoms


def run_arm(atoms: ase.Atoms, dyn, dt_fs: float, n_steps: int, warm: int, x_every: int):
    vs, xs, ts = [], [], []
    # positions are kept unwrapped: ASE's integrators move atoms without wrapping, so displacements are continuous
    for _ in range(warm):
        dyn.run(1)
    t0 = time.perf_counter()
    for i in range(n_steps):
        dyn.run(1)
        vs.append(atoms.get_velocities().astype(np.float32))
        if i % x_every == 0:
            xs.append(atoms.get_positions().astype(np.float32))
            ts.append((i + 1) * dt_fs)
    wall = time.perf_counter() - t0
    return np.asarray(vs), np.asarray(xs), np.asarray(ts), wall


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="one of our trajectories (data/trajectories/<run>.h5)")
    ap.add_argument("--start", type=int, default=9000)
    ap.add_argument("--mlip", default="pet-omatpes-v2")
    ap.add_argument("--strides", default="16,32,64", help="FlashMD strides, fs")
    ap.add_argument("--symplectic", action="store_true", help="their symplectic models (pet-omatpes: 2 and 16 fs)")
    ap.add_argument("--ref-dt", type=float, default=1.0, help="the reference MLIP's step, fs")
    ap.add_argument("--ps", type=float, default=20.0)
    ap.add_argument("--warm-steps", type=int, default=20, help="untimed steps per arm (compile, caches)")
    ap.add_argument("--x-fs", type=float, default=64.0, help="position sampling, fs (RDF, diffusion)")
    ap.add_argument("--arms", default="reference,flashmd")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    report = {"run": args.run, "start": args.start, "mlip": args.mlip, "ps": args.ps, "ref_dt_fs": args.ref_dt,
              "arms": {}}
    path = OUT / f"flashmd_{args.run}_{args.tag}.json"
    if path.exists():
        old = json.loads(path.read_text())
        if (old["run"], old["start"], old["mlip"], old["ps"]) != (args.run, args.start, args.mlip, args.ps):
            raise SystemExit(f"{path} was written with other settings; pass another --tag")
        report["arms"] = old["arms"]
    strides = [int(s) for s in args.strides.split(",") if s]
    energy_model = get_pretrained(args.mlip, strides[0] if strides else 16, symplectic=args.symplectic)[0]
    arms = args.arms.split(",")
    plan = ([("reference", args.ref_dt)] if "reference" in arms else []) + \
           ([(f"flashmd_{s}{'s' if args.symplectic else ''}", float(s)) for s in strides] if "flashmd" in arms else [])
    for name, dt_fs in plan:
        atoms = start_atoms(args.run, args.start)
        atoms.calc = EnergyCalculator(energy_model, device="cuda")
        if name == "reference":
            dyn = AseVerlet(atoms, timestep=dt_fs * units.fs)
        else:
            # symplectic: (MLIP, stride model, symplectic model); their integrator takes the last two as a pair
            loaded = get_pretrained(args.mlip, int(dt_fs), symplectic=args.symplectic)
            model = (loaded[1], loaded[2]) if args.symplectic else loaded[1]
            dyn = FlashVerlet(atoms, timestep=dt_fs * units.fs, model=model, device="cuda")
        n_steps = round(args.ps * 1000.0 / dt_fs)
        x_every = max(1, round(args.x_fs / dt_fs))
        e0 = atoms.get_potential_energy() + atoms.get_kinetic_energy()
        vs, xs, ts, wall = run_arm(atoms, dyn, dt_fs, n_steps, args.warm_steps, x_every)
        e1 = atoms.get_potential_energy() + atoms.get_kinetic_energy()
        n = len(atoms)
        row = {"dt_fs": dt_fs, "steps": n_steps, "symplectic": bool(args.symplectic and name != "reference"),
               "wall_s": wall, "ms_per_fs": 1e3 * wall / (n_steps * dt_fs),
               "energy_change_meV_per_atom": 1e3 * (e1 - e0) / n, "finite": bool(np.isfinite(xs).all())}
        report["arms"][name] = row
        np.savez(OUT / f"flashmd_{args.run}_{args.tag}_{name}.npz", v=vs, x=xs, t=ts, v_fs=dt_fs,
                 x_fs=dt_fs * x_every, numbers=atoms.get_atomic_numbers(), cell=np.asarray(atoms.get_cell()),
                 masses=atoms.get_masses())
        print(f"{name:12s} dt {dt_fs:5.1f} fs | {row['ms_per_fs']:8.3f} ms per fs | energy change "
              f"{row['energy_change_meV_per_atom']:+.3f} meV/atom over {args.ps:g} ps | finite {row['finite']}", flush=True)
        path.write_text(json.dumps(report, indent=1))
    if "reference" in report["arms"]:
        ref = report["arms"]["reference"]["ms_per_fs"]
        for row in report["arms"].values():
            row["speedup_vs_reference"] = ref / row["ms_per_fs"]
        path.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: round(v.get("speedup_vs_reference", 1.0), 2) for k, v in report["arms"].items()}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
