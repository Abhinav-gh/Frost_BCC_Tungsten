"""How fast does the lagged-teacher pipeline heat without friction? (new_directions.md, C2)

A one-step lag is negative friction: on a mode it grows the energy at a rate
g = mu theta sin(theta) / D. Under a Langevin thermostat of friction gamma this
shows up as a temperature excess, T_eff / T = gamma / (gamma - g); in NVE, with no
friction, as exponential heating at rate g. The two must agree, with nothing
fitted. On Li3PO4 the NVT runs gave +4.0 K at gamma = 0.01 fs^-1 and +1.5 K at
0.02, i.e. g = 1.3e-4 and 1.0e-4 per fs.

This runs the async pipeline in NVE from the same frame and records the kinetic
temperature at every outer step, stopping cleanly at 3x the starting temperature,
at a non-finite value, or at --fs. The synchronous scheme runs as a control for
the same duration. The heating rate is fitted to ln(T - T_eq) over the rising part.

Output: data/results/e9/c2_nve_growth_<run>.json

Usage (guarded launcher; both GPUs)
  scripts/safe_run.sh c2_nve_growth .venvs/frost-mace/bin/python \\
      experiments/e9_direction/c2_nve_growth.py --run li3po4_T300 \\
      --student data/models/students/li3po4_T300_s35c16e1.pt --k 6
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TRAJ = ROOT / "data" / "trajectories"
OUT = ROOT / "data" / "results" / "e9"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="li3po4_T300")
    ap.add_argument("--student", required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--fs", type=int, default=8000)
    ap.add_argument("--start", type=int, default=9000)
    ap.add_argument("--teacher-device", default="cuda:1")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    from mace.calculators import mace_mp

    from frost.async_teacher import LaggedResidual, MaceTeacherSpec, ProcessTeacher
    from frost.integrators import Integrator, MDState
    from frost.neighbours import VerletList
    from frost.student import GraphedStudent
    from frost.students import load_student
    from frost.teacher import frame_atoms
    from frost.teacher_fast import GraphedTeacher

    dev = "cuda:0"
    with h5py.File(TRAJ / f"{args.run}.h5", "r") as fh:
        numbers = np.asarray(fh.attrs["numbers"], dtype=np.int64)
        cell = np.asarray(fh.attrs["cell"], dtype=np.float64)
        positions = np.asarray(fh["positions"][args.start], dtype=np.float64)
        velocities = np.asarray(fh["velocities"][args.start], dtype=np.float64)
    atoms = frame_atoms(numbers, positions, cell)
    n = len(atoms)
    masses = torch.as_tensor(atoms.get_masses(), dtype=torch.float64, device=dev).unsqueeze(1)
    x0 = torch.as_tensor(atoms.get_positions(), dtype=torch.float64, device=dev)
    v0 = torch.as_tensor(velocities, dtype=torch.float64, device=dev)

    remote = ProcessTeacher(MaceTeacherSpec(atoms, cell), visible_device=args.teacher_device.split(":")[-1])
    calc = mace_mp(model="medium", device=dev, default_dtype="float32", enable_cueq=True)
    teacher = GraphedTeacher(calc, atoms, dev)
    t_list = VerletList(cell, teacher.r_max, 1.0, dev, wrap_positions=False, builder="alchemi")
    t_list.update(x0, force=True)
    teacher.set_edges(t_list.i, t_list.j, t_list.shifts, t_list.unit_shifts)
    student = load_student(args.student, dev)
    if student.config.get("backend", "e3nn") != "cueq":
        student = student.converted("cueq")
    types_t = student.types(torch.as_tensor(numbers, device=dev))
    s_list = VerletList(cell, float(student.config["cutoff"]), 0.3, dev, wrap_positions=False,
                        builder="alchemi")
    s_list.update(x0, force=True)
    graphed = GraphedStudent(student, types_t, n, capacity=int(1.3 * s_list.i.numel()) + 64)
    graphed.set_edges(s_list.i, s_list.j, s_list.shifts)

    def teacher_force(x):
        if t_list.update(x)[1]:
            teacher.set_edges(t_list.i, t_list.j, t_list.shifts, t_list.unit_shifts)
        e, f = teacher(x)
        return float(e), f.double()

    def student_force(x):
        if s_list.update(x)[1]:
            graphed.set_edges(s_list.i, s_list.j, s_list.shifts)
        e, f = graphed(x)
        return e.detach().sum(), f.detach().double(), x

    steps = round(args.fs / args.dt)
    report = {"run": args.run, "k": args.k, "dt_fs": args.dt, "student": args.student}
    for scheme in ("async", "sync"):
        state = MDState(x0.clone(), v0.clone(), masses)
        integ = Integrator(state, args.dt, None)
        _, f_t = teacher_force(state.x)
        _, f_s, _ = student_force(state.x)
        residual = f_t - f_s
        t0 = state.temperature()
        trace = [(0.0, t0)]
        lag = LaggedResidual(remote, residual) if scheme == "async" else None
        if lag is not None:
            integ.kick(residual, 0.5 * args.k * integ.dt)
        step, stopped = 0, "time"
        while step < steps:
            if lag is not None:
                for _ in range(args.k):
                    _, f_s = integ.inner_step(student_force, f_s)
                step += args.k
                kick, _ = lag.boundary(state.x, f_s, state.kinetic_energy(), step)
                integ.kick(kick, args.k * integ.dt)
            else:
                _, f_s = integ.respa_outer(student_force, f_s, residual, args.k)
                _, f_t = teacher_force(state.x)
                residual = f_t - f_s
                integ.kick(residual, 0.5 * args.k * integ.dt)
                step += args.k
            temp = state.temperature()
            trace.append((step * args.dt, temp))
            if not math.isfinite(temp):
                stopped = "non-finite"
                break
            if temp > 3.0 * t0:
                stopped = "3x temperature"
                break
        if lag is not None:
            try:
                lag.drain()
            except RuntimeError:
                pass
        t = np.array([p[0] for p in trace if math.isfinite(p[1])])
        temps = np.array([p[1] for p in trace if math.isfinite(p[1])])
        # equilibrium temperature: the first picosecond's mean (equipartition settles in ~100 fs)
        t_eq = float(temps[(t > 100) & (t < 1000)].mean())
        excess = temps - t_eq
        rising = (t > 1000) & (excess > 0.02 * t_eq)
        rate = None
        if rising.sum() >= 10:
            rate = float(np.polyfit(t[rising], np.log(excess[rising]), 1)[0])  # per fs
        report[scheme] = {"stopped": stopped, "fs_run": float(t[-1]), "T0": t0, "T_eq_first_ps": t_eq,
                          "T_final": float(temps[-1]), "growth_rate_per_fs": rate,
                          "trace_every_10_outer": [(float(a), float(b)) for a, b in trace[::10]]}
        print(f"{scheme}: stopped by {stopped} at {t[-1]:.0f} fs | T_eq {t_eq:.1f} K -> "
              f"{temps[-1]:.1f} K | fitted heating rate "
              f"{rate if rate is None else f'{rate:.2e}'} per fs", flush=True)
    remote.close()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"c2_nve_growth_{args.run}.json"
    path.write_text(json.dumps(report, indent=1))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
