"""Where does an eager inner step's time go? (MD e2e profile, liquid Cu)

The async liquid-Cu run spends 2.78 ms per inner step against 1.87 ms for the
student's force call alone. This times 240 inner steps of the production path
(``Integrator.inner_step`` with the student on its Verlet list, as ``validate.py``
runs them), switching one piece off at a time:
- **full**: list update every step (host sync), sampling every step (D2H), integrator;
- **no sampling**: drop the per-step ``.cpu()`` of x and v;
- **no list check**: the list built once and held (an upper bound on what avoiding
  the per-step sync could save; the trajectory is short enough to stay inside the skin
  is *not* assumed, this is timing only);
- **force only**: the student call alone, k times.

Also counts list rebuilds per step and times one rebuild.

Usage (guarded launcher; GPU)
  scripts/safe_run.sh inner_profile .venvs/frost-mace/bin/python \\
      experiments/e6_dynamics/inner_step_profile.py
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
    ap.add_argument("--dt", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--skin", type=float, default=0.3)
    ap.add_argument("--compile", default="", help="CompiledGraphedStudent mode (as validate.py --student-compile)")
    ap.add_argument("--start", type=int, default=5000)
    args = ap.parse_args()

    from frost.integrators import Integrator, Langevin, MDState
    from frost.neighbours import VerletList
    from frost.student import GraphedStudent
    from frost.students import load_student
    from frost.teacher import frame_atoms

    dev = "cuda"
    with h5py.File(ROOT / "data" / "trajectories" / f"{args.run}.h5", "r") as fh:
        numbers = np.asarray(fh.attrs["numbers"], dtype=np.int64)
        cell = np.asarray(fh.attrs["cell"], dtype=np.float64)
        pos = np.asarray(fh["positions"][args.start], dtype=np.float64)
        vel = np.asarray(fh["velocities"][args.start], dtype=np.float64)
    atoms = frame_atoms(numbers, pos, cell)
    n = len(atoms)
    masses = torch.as_tensor(atoms.get_masses(), dtype=torch.float64, device=dev).unsqueeze(1)
    x0 = torch.as_tensor(pos, dtype=torch.float64, device=dev)
    v0 = torch.as_tensor(vel, dtype=torch.float64, device=dev)
    temperature = float(args.run.split("_T")[1])
    student = load_student(ROOT / args.student, dev)
    types = student.types(torch.as_tensor(numbers, device=dev))
    sl = VerletList(cell, float(student.config["cutoff"]), args.skin, dev, wrap_positions=False,
                    builder="alchemi")
    sl.update(x0, force=True)
    gs = GraphedStudent(student, types, n, capacity=int(1.5 * sl.i.numel()) + 64)
    gs.set_edges(sl.i, sl.j, sl.shifts)
    if args.compile:
        from frost.student_fast import CompiledGraphedStudent

        gs = CompiledGraphedStudent(student, types, n, capacity=int(1.5 * sl.i.numel()) + 64, mode=args.compile)
        gs.set_edges(sl.i, sl.j, sl.shifts)

    def force(check: bool):
        def f(x):
            if check and sl.update(x)[1]:
                gs.set_edges(sl.i, sl.j, sl.shifts)
            e, ff = gs(x)
            return e.detach().sum(), ff.detach().double(), x
        return f

    def run(check: bool, sample: bool) -> tuple[float, int]:
        sl.update(x0, force=True)
        gs.set_edges(sl.i, sl.j, sl.shifts)
        before = sl.rebuilds
        state = MDState(x0.clone(), v0.clone(), masses)
        integ = Integrator(state, args.dt, Langevin(args.dt, 0.01, temperature, masses, seed=3))
        fn = force(check)
        f = fn(state.x)[1].clone()
        store = []
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.steps):
            _, f = integ.inner_step(fn, f)
            if sample:
                store.append(state.v.to(torch.float32).cpu())
                store.append(state.x.to(torch.float32).cpu())
        torch.cuda.synchronize()
        return 1e3 * (time.perf_counter() - t0) / args.steps, sl.rebuilds - before

    # the derived list: same edges as a cell-list build, from the teacher-radius outer list
    outer = VerletList(cell, 6.0, 1.0, dev, wrap_positions=False, builder="alchemi")
    derived = VerletList(cell, float(student.config["cutoff"]), args.skin, dev, wrap_positions=False,
                         builder="alchemi", outer=outer)
    xs = x0 + 0.05 * torch.randn_like(x0)
    outer.update(x0, force=True)
    sl.update(xs, force=True)
    derived.update(xs, force=True)

    def edge_set(lst):
        return set(zip(lst.i.tolist(), lst.j.tolist(), map(tuple, lst.unit_shifts.long().tolist())))

    same = edge_set(sl) == edge_set(derived)
    print(f"derived list == cell-list build: {same} ({sl.i.numel()} edges)", flush=True)
    t0 = time.perf_counter()
    for _ in range(20):
        derived.update(xs, force=True)
    torch.cuda.synchronize()
    print(f"one derived rebuild: {1e3 * (time.perf_counter() - t0) / 20:.3f} ms", flush=True)
    cell_list = sl
    sl = derived  # the remaining timings use the derived list

    run(True, True)  # warm
    rows = {}
    for name, check, sample in (("full", True, True), ("no sampling", True, False),
                                ("no list check", False, True), ("neither", False, False)):
        ms, rb = min(run(check, sample) for _ in range(3))
        rows[name] = (ms, rb)
        print(f"{name:14s}: {ms:.3f} ms per inner step | list rebuilds {rb} in {args.steps} steps", flush=True)
    x = x0.clone()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        gs(x)
    torch.cuda.synchronize()
    print(f"force only    : {1e3 * (time.perf_counter() - t0) / 50:.3f} ms per call", flush=True)
    t0 = time.perf_counter()
    for _ in range(20):
        sl.update(x0, force=True)
    torch.cuda.synchronize()
    print(f"one rebuild   : {1e3 * (time.perf_counter() - t0) / 20:.3f} ms", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
