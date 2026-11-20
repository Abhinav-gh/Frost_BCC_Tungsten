"""Post-mortem of a multirate blow-up saved by validate.py (liq50, 2026-10-01).

``validate.py`` keeps the last eight outer-boundary frames of a frost arm whose
energy goes non-finite. For each frame this reports:
- the minimum pair distance (minimum image) and the closest pair;
- teacher and student forces on that pair's atoms, and the residual R = F_T - F_S
  there and overall (max and RMS);
- the kinetic temperature.

These are set against the teacher trajectory the student was trained on: its
minimum pair distance over all frames, and its largest force.

The question is whether the blow-up starts where the student was never trained,
a close approach it under-repels, with the impulse kick then amplifying it.

Usage (GPU; small)
  LANE=check scripts/safe_run.sh postmortem .venvs/frost-mace/bin/python \\
      experiments/e6_dynamics/blowup_postmortem.py data/results/e6/blowup_<...>.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TRAJ = ROOT / "data" / "trajectories"


def min_pair(x: np.ndarray, cell: np.ndarray):
    inv = np.linalg.inv(cell)
    d = x[:, None, :] - x[None, :, :]
    frac = d @ inv
    frac -= np.round(frac)
    r = np.linalg.norm(frac @ cell, axis=-1)
    np.fill_diagonal(r, np.inf)
    i, j = np.unravel_index(np.argmin(r), r.shape)
    return float(r[i, j]), int(i), int(j)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("frames", type=Path)
    ap.add_argument("--run", default="cu500_T1600")
    ap.add_argument("--student", default="data/models/students/cu500_T1600_s35c16e1liq.pt")
    ap.add_argument("--train-stride", type=int, default=50)
    args = ap.parse_args()

    from mace.calculators import mace_mp

    from frost.neighbours import VerletList
    from frost.student import energy_and_forces
    from frost.students import load_student
    from frost.trajectory import frame_atoms

    blob = torch.load(args.frames, map_location="cpu")
    with h5py.File(TRAJ / f"{args.run}.h5", "r") as fh:
        numbers = np.asarray(fh.attrs["numbers"], dtype=np.int64)
        cell = np.asarray(fh.attrs["cell"], dtype=np.float64)
        train_x = np.asarray(fh["positions"][:: args.train_stride], dtype=np.float64)
    train_min = min(min_pair(x, cell)[0] for x in train_x)
    calc = mace_mp(model="medium", device="cuda", default_dtype="float32")
    student = load_student(ROOT / args.student, "cuda")
    types = student.types(torch.as_tensor(numbers, device="cuda"))

    def student_forces(x: np.ndarray) -> np.ndarray:
        xt = torch.as_tensor(x, dtype=torch.float64, device="cuda")
        nl = VerletList(cell, float(student.config["cutoff"]), 0.0, "cuda", wrap_positions=False,
                        builder="vesin")
        nl.update(xt, force=True)
        _, f = energy_and_forces(student, xt.float(), types, nl.i, nl.j, nl.shifts.float())
        return f.detach().double().cpu().numpy()
    masses = frame_atoms(numbers, train_x[0], cell).get_masses()
    rows = []
    for step, x, v in zip(blob["steps"], blob["x"], blob["v"]):
        x = x.double().numpy()
        v = v.double().numpy()
        if not np.all(np.isfinite(x)):
            rows.append({"step": step, "finite": False})
            continue
        atoms = frame_atoms(numbers, x, cell)
        atoms.calc = calc
        f_t = atoms.get_forces()
        f_s = student_forces(x)
        r, i, j = min_pair(x, cell)
        kin = 0.5 * np.sum(masses[:, None] * v**2) * 103.6427  # amu A^2/fs^2 -> eV
        row = {"step": step, "t_fs": step * blob["dt_fs"], "finite": True, "min_pair_A": r,
               "pair": [i, j], "temperature_K": 2 * kin / (3 * len(x) * 8.617e-5),
               "max_teacher_force": float(np.abs(f_t).max()),
               "teacher_force_on_pair": float(np.linalg.norm(f_t[[i, j]], axis=1).max())}
        if f_s is not None:
            res = f_t - f_s
            row.update({"residual_max": float(np.linalg.norm(res, axis=1).max()),
                        "residual_rms": float(np.sqrt(np.mean(np.sum(res**2, axis=1)))),
                        "residual_on_pair": float(np.linalg.norm(res[[i, j]], axis=1).max())})
        rows.append(row)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}),
              flush=True)
    print(f"teacher trajectory: min pair distance {train_min:.3f} A over "
          f"{len(train_x)} frames (stride {args.train_stride})", flush=True)
    out = args.frames.with_suffix(".postmortem.json")
    out.write_text(json.dumps({"frames": rows, "train_min_pair_A": train_min}, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
