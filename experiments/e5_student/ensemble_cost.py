"""Does ensemble batching raise the multirate speedup? A cost check before a harness (test 14).

new_directions.md M3: when the job is an ensemble (replicas, seeds, windows), the
second GPU is better used by data parallelism than by the lagged pipeline, and the
gain left is batching. The student is launch-bound (1.02 ms device time inside a
launch-bound call), the teacher compute-bound. So batching M replicas should cost
the student little and the teacher about M times. The multirate speedup at a fixed
k would then rise with M.

This measures the premise before any harness is built. M replicas are timed as an
M-fold supercell, which has the same atom and edge counts as M disjoint copies:
- the teacher (MACE-MP-0 medium, cuEquivariance, CUDA graph) per call;
- the student (``--student``, cuEquivariance, CUDA graph) per call.

The R3 cost model then predicts the sync speedup per M:

    S(M) = k c_T(M) / (k c_S(M) + c_T(M) + k o),

where o is the per-step integrator overhead fitted to the measured M = 1 run.

Output: data/results/e5/ensemble_cost_<run>.json

Usage (guarded launcher; GPU)
  scripts/safe_run.sh ensemble_cost .venvs/frost-mace/bin/python \\
      experiments/e5_student/ensemble_cost.py --run li3po4_T300 \\
      --student data/models/students/li3po4_T300_s35c16e1.pt --k 6 --measured-speedup 3.66
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
REPEATS = {1: (1, 1, 1), 2: (2, 1, 1), 4: (2, 2, 1), 8: (2, 2, 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="li3po4_T300")
    ap.add_argument("--student", default="data/models/students/li3po4_T300_s35c16e1.pt")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--measured-speedup", type=float, default=3.66,
                    help="the measured sync speedup at M = 1, which fixes the overhead o")
    ap.add_argument("--replicas", default="1,2,4,8")
    args = ap.parse_args()

    from mace.calculators import mace_mp

    from frost.neighbours import VerletList
    from frost.student import GraphedStudent
    from frost.students import load_student
    from frost.teacher import frame_atoms, time_call_ms
    from frost.teacher_fast import GraphedTeacher

    dev = "cuda"
    with h5py.File(ROOT / "data" / "trajectories" / f"{args.run}.h5", "r") as fh:
        numbers = np.asarray(fh.attrs["numbers"], dtype=np.int64)
        cell = np.asarray(fh.attrs["cell"], dtype=np.float64)
        positions = np.asarray(fh["positions"][0], dtype=np.float64)
    base = frame_atoms(numbers, positions, cell)
    calc = mace_mp(model="medium", device=dev, default_dtype="float32", enable_cueq=True)
    student = load_student(ROOT / args.student, dev)
    if student.config.get("backend", "e3nn") != "cueq":
        student = student.converted("cueq")

    rows = []
    for m in (int(x) for x in args.replicas.split(",")):
        atoms = base.repeat(REPEATS[m])
        x = torch.as_tensor(atoms.get_positions(), dtype=torch.float64, device=dev)
        big_cell = np.asarray(atoms.get_cell())
        teacher = GraphedTeacher(calc, atoms, dev)
        tl = VerletList(big_cell, teacher.r_max, 1.0, dev, wrap_positions=False, builder="alchemi")
        tl.update(x, force=True)
        teacher.set_edges(tl.i, tl.j, tl.shifts, tl.unit_shifts)
        c_t = time_call_ms(lambda: teacher(x), repeats=20)
        types = student.types(torch.as_tensor(atoms.get_atomic_numbers(), device=dev))
        sl = VerletList(big_cell, float(student.config["cutoff"]), 0.3, dev, wrap_positions=False,
                        builder="alchemi")
        sl.update(x, force=True)
        gs = GraphedStudent(student, types, len(atoms), capacity=int(1.3 * sl.i.numel()) + 64)
        gs.set_edges(sl.i, sl.j, sl.shifts)
        c_s = time_call_ms(lambda: gs(x), repeats=50)
        rows.append({"replicas": m, "atoms": len(atoms), "teacher_ms": c_t, "student_ms": c_s,
                     "teacher_ms_per_replica": c_t / m, "student_ms_per_replica": c_s / m})
        print(f"M {m}: {len(atoms)} atoms | teacher {c_t:.2f} ms ({c_t / m:.2f}/replica) | student "
              f"{c_s:.3f} ms ({c_s / m:.3f}/replica) | ratio {c_t / c_s:.1f}", flush=True)
        del teacher, gs
        torch.cuda.empty_cache()

    k, s1 = args.k, args.measured_speedup
    r1 = rows[0]
    # S(1) = k c_T / (k c_S + c_T + k o)  ->  o
    o = (k * r1["teacher_ms"] / s1 - k * r1["student_ms"] - r1["teacher_ms"]) / k
    for r in rows:
        # the per-step overhead is integrator and list work on M times the atoms: bounded
        # below by o (launch-bound) and above by M o (bandwidth-bound); both are reported
        for name, ov in (("pred_speedup_overhead_flat", o), ("pred_speedup_overhead_linear", o * r["replicas"])):
            r[name] = k * r["teacher_ms"] / (k * r["student_ms"] + r["teacher_ms"] + k * ov)
        print(f"M {r['replicas']}: predicted sync speedup at k = {k}: "
              f"{r['pred_speedup_overhead_flat']:.2f}x (overhead flat) to "
              f"{r['pred_speedup_overhead_linear']:.2f}x (overhead x M)", flush=True)
    out = ROOT / "data" / "results" / "e5" / f"ensemble_cost_{args.run}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"k": k, "measured_speedup_M1": s1, "overhead_ms_per_step": o,
                               "rows": rows}, indent=1))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
