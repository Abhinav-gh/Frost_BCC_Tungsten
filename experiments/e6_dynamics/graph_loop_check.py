"""GraphedLoop against the eager integrator: same trajectory, and how much faster.

From one liquid-Cu frame and one Langevin seed, k inner steps are run four ways:
- the student force, eager (``Integrator.inner_step`` with ``GraphedStudent``, as
  ``validate.py`` runs it);
- the student force, as one ``GraphedLoop`` replay;
- the same pair with the teacher force.

Reported: the largest position and velocity differences after k steps (they
should sit at float32 force round-off), and ms per inner step for each path, best
of several repeats.

Usage (guarded launcher; GPU)
  scripts/safe_run.sh graph_loop_check .venvs/frost-mace/bin/python \\
      experiments/e6_dynamics/graph_loop_check.py --run cu500_T1600 \\
      --student data/models/students/cu500_T1600_s35c16e1liq.pt --k 12 --dt 2
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="cu500_T1600")
    ap.add_argument("--student", default="data/models/students/cu500_T1600_s35c16e1liq.pt")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--dt", type=float, default=2.0)
    ap.add_argument("--student-skin", type=float, default=1.0)
    args = ap.parse_args()

    from mace.calculators import mace_mp

    from frost.graph_loop import GraphedLoop, StudentForce, TeacherForce
    from frost.integrators import Integrator, Langevin, MDState
    from frost.neighbours import VerletList
    from frost.student import GraphedStudent
    from frost.students import load_student
    from frost.teacher import frame_atoms
    from frost.teacher_fast import GraphedTeacher

    dev = "cuda"
    with h5py.File(ROOT / "data" / "trajectories" / f"{args.run}.h5", "r") as fh:
        numbers = np.asarray(fh.attrs["numbers"], dtype=np.int64)
        cell = np.asarray(fh.attrs["cell"], dtype=np.float64)
        pos = np.asarray(fh["positions"][5000], dtype=np.float64)
        vel = np.asarray(fh["velocities"][5000], dtype=np.float64)
    atoms = frame_atoms(numbers, pos, cell)
    n = len(atoms)
    masses = torch.as_tensor(atoms.get_masses(), dtype=torch.float64, device=dev).unsqueeze(1)
    x0 = torch.as_tensor(pos, dtype=torch.float64, device=dev)
    v0 = torch.as_tensor(vel, dtype=torch.float64, device=dev)
    temperature = float(args.run.split("_T")[1])

    calc = mace_mp(model="medium", device=dev, default_dtype="float32", enable_cueq=True)
    teacher = GraphedTeacher(calc, atoms, dev)
    tl = VerletList(cell, teacher.r_max, 1.0, dev, wrap_positions=False, builder="alchemi")
    tl.update(x0, force=True)
    teacher.set_edges(tl.i, tl.j, tl.shifts, tl.unit_shifts)
    student = load_student(ROOT / args.student, dev)
    if student.config.get("backend", "e3nn") != "cueq":
        student = student.converted("cueq")
    types = student.types(torch.as_tensor(numbers, device=dev))
    sl = VerletList(cell, float(student.config["cutoff"]), args.student_skin, dev,
                    wrap_positions=False, builder="alchemi")
    sl.update(x0, force=True)
    gs = GraphedStudent(student, types, n, capacity=int(1.3 * sl.i.numel()) + 64)
    gs.set_edges(sl.i, sl.j, sl.shifts)

    def eager_student(x):
        e, f = gs(x)
        return e.detach().sum(), f.detach().double(), x

    def eager_teacher(x):
        e, f = teacher(x)
        return e, f.double(), x

    sf = StudentForce(student, types, n, capacity=int(1.3 * sl.i.numel()) + 64)
    sf.set_edges(sl.i, sl.j, sl.shifts)
    tf = TeacherForce(teacher)
    for name, eager_fn, force in (("student", eager_student, sf), ("teacher", eager_teacher, tf)):
        f0 = eager_fn(x0)[1].clone()
        # eager: k steps, lists held fixed (the comparison is of the integration, not the lists)
        state = MDState(x0.clone(), v0.clone(), masses)
        integ = Integrator(state, args.dt, Langevin(args.dt, 0.01, temperature, masses, seed=7))
        f = f0.clone()
        for _ in range(args.k):
            _, f = integ.inner_step(eager_fn, f)
        xe, ve = state.x.clone(), state.v.clone()
        loop = GraphedLoop(force, masses, integ.dt, args.k, Langevin(args.dt, 0.01, temperature, masses, seed=7))
        xg, vg, _ = loop.run(x0, v0, f0, sl.x_ref if name == "student" else tl.x_ref)
        dx = float((xg - xe).abs().max())
        dv = float((vg - ve).abs().max())

        def time_eager():
            st = MDState(x0.clone(), v0.clone(), masses)
            it = Integrator(st, args.dt, Langevin(args.dt, 0.01, temperature, masses, seed=7))
            ff = f0.clone()
            for _ in range(args.k):
                _, ff = it.inner_step(eager_fn, ff)
            float(st.x.sum())  # the eager loop's own end-of-interval sync

        def time_graph():
            loop.run(x0, v0, f0, sl.x_ref if name == "student" else tl.x_ref)
            float(loop.maxdisp)  # the one sync per interval

        res = {}
        for label, fn in (("eager", time_eager), ("graph", time_graph)):
            fn()
            torch.cuda.synchronize()
            best = 1e9
            for _ in range(5):
                t0 = time.perf_counter()
                for _ in range(10):
                    fn()
                torch.cuda.synchronize()
                best = min(best, (time.perf_counter() - t0) / 10)
            res[label] = 1e3 * best / args.k
        print(f"{name}: max |dx| {dx:.2e} A, max |dv| {dv:.2e} | ms per inner step: eager "
              f"{res['eager']:.3f}, graph {res['graph']:.3f} ({res['eager'] / res['graph']:.2f}x) | "
              f"max displacement over the interval {float(loop.maxdisp):.3f} A", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
