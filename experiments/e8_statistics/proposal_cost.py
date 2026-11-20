"""E8 profile: where does a nested-HMC proposal's wall time go?

The statistics contract pays for one teacher *energy* call per proposal and
``n_draft`` student force calls. Which of those dominates decides what to
optimise, and the answer is not obvious: the energy-only forward does a third of
the teacher's arithmetic but launches the same hundreds of small kernels, with no
backward pass to amortise them over, so eager it can cost nearly as much as a
CUDA-graph force call.

This times each piece separately and against the whole proposal, so the parts and
the measured total can be reconciled:

    proposal = n_draft * (student force + integrator + list check) + teacher energy

It also times the teacher's graphed force call, which is what the baseline arm
pays per femtosecond, so the ratio of femtoseconds per second between the two
arms is reported next to it. That ratio is the ceiling on S_stat; acceptance and
the autocorrelation time (``sample.py``) decide how much of it is realised.

Output: data/results/e8/proposal_cost_<run>[_<tag>].json
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
OUT_DIR = ROOT / "data" / "results" / "e8"


def timed(call, repeats: int, warmup: int = 5) -> float:
    """Median seconds per call, synchronised."""
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        call()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def main() -> int:
    import h5py
    from mace.calculators import mace_mp

    from frost.neighbours import VerletList
    from frost.nested import NestedHMC
    from frost.student import GraphedStudent
    from frost.students import load_student
    from frost.teacher import frame_atoms
    from frost.teacher_fast import GraphedTeacher

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="silica1152_T300")
    ap.add_argument("--student", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--start", type=int, default=9000)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--n-draft", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dev = args.device
    temperature = float(args.run.split("_T")[1])

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

    parts = {
        "teacher_force_graph_ms": 1e3 * timed(lambda: teacher(x0), args.repeats),
        "teacher_energy_eager_ms": 1e3 * timed(lambda: teacher.energy_eager(x0), args.repeats),
        "teacher_energy_graph_ms": 1e3 * timed(lambda: teacher.energy(x0), args.repeats),
        "student_force_graph_ms": 1e3 * timed(lambda: graphed(x0), args.repeats),
        "student_force_wrapped_ms": 1e3 * timed(lambda: student_force(x0), args.repeats),
    }

    sampler = NestedHMC(masses, temperature, dt_fs=args.dt, n_draft=args.n_draft,
                        student_force=student_force, teacher_energy=teacher_energy, seed=0)
    x, v = x0.clone(), torch.zeros_like(x0)
    e_t = float(teacher_energy(x))
    f_s = student_force(x)[1]
    state = {"x": x, "v": v, "e": e_t, "f": f_s}

    def one_proposal():
        out = sampler.step(state["x"], state["v"], state["e"], state["f"])
        state["x"], state["v"], state["e"], state["f"] = out[:4]

    parts["proposal_ms"] = 1e3 * timed(one_proposal, args.repeats, warmup=3)

    drafted = args.n_draft * parts["student_force_wrapped_ms"]
    accounted = drafted + parts["teacher_energy_graph_ms"]
    report = {
        "run": args.run, "n_atoms": n, "student": student.config,
        "config": vars(args), "parts_ms": parts,
        "share": {
            "student_drafting": drafted / parts["proposal_ms"],
            "teacher_energy": parts["teacher_energy_graph_ms"] / parts["proposal_ms"],
            "unaccounted": (parts["proposal_ms"] - accounted) / parts["proposal_ms"],
        },
        "fs_per_second": {
            "teacher_md": 1e3 / parts["teacher_force_graph_ms"],
            "nested_drafted": 1e3 * args.n_draft * args.dt / parts["proposal_ms"],
        },
    }
    report["fs_per_second"]["ratio"] = (report["fs_per_second"]["nested_drafted"]
                                        / report["fs_per_second"]["teacher_md"])
    report["energy_graph_speedup"] = (parts["teacher_energy_eager_ms"]
                                      / parts["teacher_energy_graph_ms"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out = OUT_DIR / f"proposal_cost_{args.run}{suffix}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"{args.run}: proposal {parts['proposal_ms']:.1f} ms = "
          f"{args.n_draft} x {parts['student_force_wrapped_ms']:.2f} ms student "
          f"+ {parts['teacher_energy_graph_ms']:.2f} ms teacher energy "
          f"({parts['teacher_energy_eager_ms']:.2f} eager, "
          f"{report['energy_graph_speedup']:.2f}x) | teacher MD step "
          f"{parts['teacher_force_graph_ms']:.2f} ms | drafted fs/s ratio "
          f"{report['fs_per_second']['ratio']:.2f}x", flush=True)
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
