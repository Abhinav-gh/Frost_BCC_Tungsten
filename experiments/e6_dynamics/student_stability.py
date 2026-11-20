"""E6: is the student itself stable at the inner step the multirate loop uses?

Silica's multirate arm degrades at a 2 fs inner step where its teacher does not
(plan section 1.15): halves rise rather than decelerate, and the spectrum leaves
the teacher's own spread. Two explanations survive the arithmetic, and the
harness cannot tell them apart, because both arms of a ``validate.py`` run share
the same inner step and the same true dynamics.

- The interval is fewer steps deep. H is pinned in femtoseconds by the resonance
  cap, so 6 fs is six inner steps at dt = 1 and three at dt = 2, and the residual
  is held across a trajectory integrated coarsely. This is a property of the
  multirate scheme.
- The student's own energy surface is stiffer than the teacher's, so the student
  is not stable at a step the teacher tolerates. This is a property of the
  student, and it would degrade the inner trajectory before any residual is
  applied.

Note that the boundary impulse is *not* the difference: it is 0.5 * k * dt * R =
0.5 * H * R, identical at both steps, and the measured staleness is unchanged
(0.92 against 0.91) — so neither the kick nor the residual's staleness explains it.

This runs the student alone in NVE — no teacher, no residual — and reports the
conservation of its own energy, U_S + K, at each step. A student that conserves
its own energy at 2 fs and degrades only inside the multirate loop leaves the
first explanation; one that degrades alone is the second, and the effect belongs
to the model rather than the scheme.

Output: data/results/e6/student_stability_<run>[_<tag>].json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TRAJ = ROOT / "data" / "trajectories"
OUT_DIR = ROOT / "data" / "results" / "e6"


def main() -> int:
    import h5py

    from frost.integrators import Integrator, MDState
    from frost.neighbours import VerletList
    from frost.observables import rdf
    from frost.student import GraphedStudent
    from frost.students import load_student
    from frost.teacher import frame_atoms

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="silica1152_T300")
    ap.add_argument("--student", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--start", type=int, default=9000)
    ap.add_argument("--fs", type=int, default=20000)
    ap.add_argument("--steps-fs", default="1,2,4", help="inner steps to test, fs")
    ap.add_argument("--rdf-every-fs", type=float, default=200.0,
                    help="frames kept for the student's own radial distribution. At a long "
                         "enough interval any impulse scheme tends to the student's "
                         "distribution, so this is the k = infinity row a structural gate "
                         "needs: a gate the student alone passes cannot certify Frost at any k")
    ap.add_argument("--student-skin", type=float, default=0.3)
    ap.add_argument("--neighbours", default="alchemi")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dev = args.device

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

    student = load_student(args.student, dev)
    if student.config.get("backend", "e3nn") == "e3nn" and student.config.get("kind") == "mace":
        student = student.converted("cueq")
    types = student.types(torch.as_tensor(numbers, device=dev))
    s_list = VerletList(cell, float(student.config["cutoff"]), args.student_skin, dev,
                        wrap_positions=False, builder=args.neighbours)
    s_list.update(x0, force=True)
    graphed = GraphedStudent(student, types, n, capacity=int(1.3 * s_list.i.numel()) + 64)
    graphed.set_edges(s_list.i, s_list.j, s_list.shifts)

    def student_force(x):
        if s_list.update(x)[1]:
            graphed.set_edges(s_list.i, s_list.j, s_list.shifts)
        e, f = graphed(x)
        return e.detach().sum(), f.detach().double(), x

    report = {"run": args.run, "n_atoms": n, "student": student.config,
              "config": vars(args), "steps": {}}
    for dt in [float(s) for s in args.steps_fs.split(",")]:
        state = MDState(x0.clone(), v0.clone(), masses)
        integ = Integrator(state, dt)
        energy, forces, state.x = student_force(state.x)
        trace, t_ps, frames = [], [], []
        t0 = time.perf_counter()
        for step in range(1, round(args.fs / dt) + 1):
            energy, forces = integ.inner_step(student_force, forces)
            if step % max(round(10.0 / dt), 1) == 0:  # every ~10 fs
                trace.append((float(energy) + state.kinetic_energy()) / n * 1e3)
                t_ps.append(step * dt / 1e3)
            if step % max(round(args.rdf_every_fs / dt), 1) == 0:
                frames.append(state.x.to(torch.float32).cpu())
        torch.cuda.synchronize()
        trace = np.asarray(trace) - trace[0]
        half = len(trace) // 2
        slopes = [float(np.polyfit(t_ps[a:b], trace[a:b], 1)[0]) for a, b in
                  ((0, half), (half, len(trace)))]
        report["steps"][f"{dt:g}"] = {
            "dt_fs": dt, "wall_s": time.perf_counter() - t0,
            "drift_meV_per_atom_per_ps": float(np.polyfit(t_ps, trace, 1)[0]),
            "halves_meV_per_atom_per_ps": slopes,
            "total_meV_per_atom": float(trace[-1]),
            "peak_abs_meV_per_atom": float(np.abs(trace).max()),
            "trace_meV_per_atom": [float(v) for v in trace[::max(len(trace) // 20, 1)]],
            "final_T_K": state.temperature(),
            "rdf": rdf(torch.stack(frames), cell, types.cpu().numpy(),
                       int(types.max()) + 1) if frames else None,
        }
        entry = report["steps"][f"{dt:g}"]
        print(f"{args.run} student-only dt={dt:g} fs: drift {entry['drift_meV_per_atom_per_ps']:+.4f} "
              f"meV/atom/ps | halves {slopes[0]:+.4f} / {slopes[1]:+.4f} | total "
              f"{entry['total_meV_per_atom']:+.3f} | peak |dE| {entry['peak_abs_meV_per_atom']:.3f} "
              f"| T {entry['final_T_K']:.1f} K", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out = OUT_DIR / f"student_stability_{args.run}{suffix}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
