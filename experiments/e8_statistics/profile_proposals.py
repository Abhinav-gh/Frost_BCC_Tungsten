"""E8 profile: why do nested-HMC proposals get rejected?

The replay proxy predicted 0.29-0.32 acceptance for whole-cell water proposals and
the sampler measured 0.06-0.10. The proxy used the energy residual's change along
the teacher's trajectory, so it ignored the drafts' own integration error. Every
proposal's teacher energy change splits exactly into two parts:

    dH_T = dH_S + d(U_T - U_S)

- dH_S is the student integrator's energy error along the draft (volume-preserving
  velocity Verlet does not conserve H_S exactly; light, fast hydrogens make it
  large);
- d(U_T - U_S) is the model residual's change between the draft's two ends.

This runs short chains and reports the mean and spread of all three in kT, and the
acceptance, for each (dt, n_draft) given. Configurations of equal draft length
(dt * n_draft) separate the two parts: integration error falls roughly as dt^2 at
fixed length, while the residual term does not.

Output: data/results/e8/proposals_<run>[_<tag>].json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TRAJ = ROOT / "data" / "trajectories"
OUT_DIR = ROOT / "data" / "results" / "e8"


def main() -> int:
    import h5py
    from ase import units
    from mace.calculators import mace_mp

    from frost.neighbours import VerletList
    from frost.nested import NestedHMC
    from frost.student import GraphedStudent
    from frost.students import load_student
    from frost.teacher import frame_atoms
    from frost.teacher_fast import GraphedTeacher

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="water216_T300")
    ap.add_argument("--student", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--start", type=int, default=9000)
    ap.add_argument("--configs", default="1.0x8,0.5x16,0.25x32,1.0x2",
                    help="comma list of <dt_fs>x<n_draft>")
    ap.add_argument("--proposals", type=int, default=300)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dev = args.device
    temperature = float(args.run.split("_T")[1])
    kT = units.kB * temperature

    with h5py.File(TRAJ / f"{args.run}.h5", "r") as fh:
        numbers = np.asarray(fh.attrs["numbers"], dtype=np.int64)
        cell = np.asarray(fh.attrs["cell"], dtype=np.float64)
        positions = np.asarray(fh["positions"][args.start], dtype=np.float64)
    atoms = frame_atoms(numbers, positions, cell)
    n = len(atoms)
    masses = torch.as_tensor(atoms.get_masses(), dtype=torch.float64, device=dev).unsqueeze(1)
    x0 = torch.as_tensor(atoms.get_positions(), dtype=torch.float64, device=dev)

    calc = mace_mp(model="medium", device=dev, default_dtype="float32", enable_cueq=True)
    teacher = GraphedTeacher(calc, atoms, dev)
    t_list = VerletList(cell, teacher.r_max, 1.0, dev, wrap_positions=False, builder="alchemi")
    t_list.update(x0, force=True)
    teacher.set_edges(t_list.i, t_list.j, t_list.shifts, t_list.unit_shifts)
    student = load_student(args.student, dev)
    if student.config.get("backend", "e3nn") == "e3nn" and student.config.get("kind") == "mace":
        student = student.converted("cueq")
    types = student.types(torch.as_tensor(numbers, device=dev))
    s_list = VerletList(cell, float(student.config["cutoff"]), 0.3, dev, wrap_positions=False,
                        builder="alchemi")
    s_list.update(x0, force=True)
    graphed = GraphedStudent(student, types, n, capacity=int(1.3 * s_list.i.numel()) + 64)
    graphed.set_edges(s_list.i, s_list.j, s_list.shifts)

    def teacher_energy(x):
        if t_list.update(x)[1]:
            teacher.set_edges(t_list.i, t_list.j, t_list.shifts, t_list.unit_shifts)
        return teacher.energy(x)

    def student_force(x):
        if s_list.update(x)[1]:
            graphed.set_edges(s_list.i, s_list.j, s_list.shifts)
        e, f = graphed(x)
        return e.detach().sum(), f.detach().double(), x

    report = {"run": args.run, "n_atoms": n, "student": student.config, "kT_eV": kT, "configs": {}}
    for spec in args.configs.split(","):
        dt, n_draft = float(spec.split("x")[0]), int(spec.split("x")[1])
        sampler = NestedHMC(masses, temperature, dt_fs=dt, n_draft=n_draft,
                            student_force=student_force, teacher_energy=teacher_energy, seed=7)
        x = x0.clone()
        v = torch.zeros_like(x)
        e_t = float(teacher_energy(x))
        f_s = student_force(x)[1]
        for _ in range(args.proposals):
            x, v, e_t, f_s, _ = sampler.step(x, v, e_t, f_s)
        h = sampler.history[args.proposals // 10:]
        parts = {key: np.array([row[key] for row in h]) / kT for key in ("dH_T", "dH_S", "d_residual")}
        entry = {"dt_fs": dt, "n_draft": n_draft, "draft_fs": dt * n_draft,
                 "acceptance": float(np.mean([row["accepted"] for row in h]))}
        for key, values in parts.items():
            entry[f"{key}_mean_kT"] = float(values.mean())
            entry[f"{key}_sd_kT"] = float(values.std())
        report["configs"][spec] = entry
        print(f"{args.run} dt {dt} x {n_draft} ({dt * n_draft:.0f} fs): acceptance {entry['acceptance']:.2f} | "
              f"dH_T {entry['dH_T_mean_kT']:+.2f} +/- {entry['dH_T_sd_kT']:.2f} kT | "
              f"dH_S {entry['dH_S_mean_kT']:+.2f} +/- {entry['dH_S_sd_kT']:.2f} | "
              f"d(U_T-U_S) {entry['d_residual_mean_kT']:+.2f} +/- {entry['d_residual_sd_kT']:.2f}",
              flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out = OUT_DIR / f"proposals_{args.run}{suffix}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
