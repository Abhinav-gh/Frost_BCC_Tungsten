"""E5c: how cheap can the force calls be, on both sides of the speedup?

G2 asks for a student call at most 0.05 of the teacher's, and the teacher that
counts is the fastest one we can run. On one frame and GPU this times:
- the teacher: e3nn, cuEquivariance, and cuEquivariance replayed from a CUDA
  graph (``GraphedTeacher``), which the baseline uses too;
- every saved student: eager autograd, the CUDA-graph call (``GraphedStudent``),
  Inductor-compiled kernels inside a CUDA graph and the same in bfloat16, plus,
  for MACE students, the cuEquivariance and OpenEquivariance backends inside a
  CUDA graph, and cuEquivariance under bfloat16 autocast (``AutocastStudent``);
- one in-situ training update per MACE backend;
- neighbour-list builds at the teacher's and the students' radii, with vesin on
  the CPU and ALCHEMI's cell list on the GPU.

Every accelerated force call is checked against eager e3nn forces on all atoms,
and only calls within 1e-3 eV/A count towards the best c_S. The two list
builders must agree edge for edge.

Output: data/results/e5/student_cost_<run>[_<tag>].json

Usage (guarded launcher; GPU)
  scripts/safe_run.sh e5_student_cost .venvs/frost-mace/bin/python \\
      experiments/e5_student/student_cost.py --run silica1152_T300 --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TRAJ = ROOT / "data" / "trajectories"
RESULTS = ROOT / "data" / "results" / "e5"
STUDENTS = ROOT / "data" / "models" / "students"
FORCE_TOL = 1e-3  # eV/A


def error_text(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:200]


def time_teachers(atoms, cell, device: str) -> dict:
    """The teacher's force and energy-only calls: e3nn, cuEquivariance, and cuEquivariance graphed."""
    from mace.calculators import mace_mp

    from frost.neighbours import VerletList
    from frost.teacher import MaceTeacher, mace_forces, time_call_ms, time_mace_ms
    from frost.teacher_fast import GraphedTeacher

    out = {}
    for tag, cueq in (("e3nn", False), ("cueq", True)):
        try:
            calc = mace_mp(model="medium", device=device, default_dtype="float32",
                           enable_cueq=cueq)
            _, _, batch = mace_forces(calc, atoms, device)
            out[f"teacher_{tag}_force_ms"] = time_mace_ms(calc.models[0], batch)
            out[f"teacher_{tag}_energy_ms"] = time_mace_ms(calc.models[0], batch, force=False)
            if cueq:
                # the online loop's call: a 1 A skin Verlet list, eager and graphed
                verlet = VerletList(cell, float(calc.models[0].r_max), 1.0, device,
                                    wrap_positions=False)
                x = torch.as_tensor(atoms.get_positions(), dtype=torch.float64, device=device)
                verlet.update(x, force=True)
                eager = MaceTeacher(calc, atoms, device)
                graphed = GraphedTeacher(calc, atoms, device)
                for model in (eager, graphed):
                    model.set_edges(verlet.i, verlet.j, verlet.shifts, verlet.unit_shifts)
                f_eager = eager(x)[1].clone()
                out["teacher_cueq_graph_max_abs_force_diff"] = float(
                    (graphed(x)[1] - f_eager).abs().max())
                out["teacher_cueq_skinlist_force_ms"] = time_call_ms(lambda m=eager, p=x: m(p),
                                                                     repeats=20)
                out["teacher_cueq_graph_force_ms"] = time_call_ms(lambda m=graphed, p=x: m(p),
                                                                  repeats=20)
                del eager, graphed, verlet
            del calc, batch
        except Exception as exc:  # e3nn always works; record why an accelerated path did not
            out[f"teacher_{tag}_error"] = error_text(exc)
        torch.cuda.empty_cache()
    return out


def time_train_update(path, backend: str, x, i, j, sh, numbers, device: str) -> float:
    """One in-situ update on four pairs of frames; the label values do not change its cost."""
    from frost.insitu import Frame, InSituTrainer, LabelBuffer
    from frost.students import load_student
    from frost.teacher import time_call_ms

    student = load_student(path, device).float()
    if backend != student.config.get("backend", "e3nn"):
        student = student.converted(backend)
    types = student.types(torch.as_tensor(numbers, device=device))
    buffer = LabelBuffer(16)
    for step in range(8):
        buffer.add(Frame(step, x.clone(), torch.zeros_like(x), 0.0, i, j, sh))
    trainer = InSituTrainer(student, types, lr=1e-5, max_gap=4)
    return time_call_ms(lambda: trainer.step(buffer, batch_pairs=4), repeats=10, warmup=3)


def time_variants(path, atoms, cell, numbers, device: str) -> dict:
    """Every force-call variant for one saved student, each checked against eager e3nn."""
    from frost.neighbours import edges_for
    from frost.student import GraphedStudent, energy_and_forces
    from frost.student_fast import AutocastStudent, CompiledGraphedStudent
    from frost.students import load_student
    from frost.teacher import time_call_ms

    n = len(atoms)
    student = load_student(path, device).float()
    types = student.types(torch.as_tensor(numbers, device=device))
    xw, i, j, sh, _ = edges_for(atoms.get_positions(), cell, float(student.config["cutoff"]), device)
    capacity = int(1.2 * i.numel()) + 16

    def eager():
        return energy_and_forces(student, xw, types, i, j, sh)

    _, f_ref = eager()
    # a checkpoint trained on fused kernels is its own reference; only e3nn converts
    backend = student.config.get("backend", "e3nn")
    row: dict = {"config": student.config, "edges": int(i.numel()), "reference_backend": backend}
    row["eager_ms"] = time_call_ms(eager, repeats=20)

    builders = {
        "graph": lambda: GraphedStudent(student, types, n, capacity),
        "compiled_graph": lambda: CompiledGraphedStudent(student, types, n, capacity),
    }
    if student.config.get("kind") == "mace":
        if backend == "e3nn":
            for other in ("cueq", "oeq"):
                builders[f"graph_{other}"] = (
                    lambda b=other: GraphedStudent(student.converted(b), types, n, capacity))
        builders["graph_cueq_autocast_bf16"] = lambda: GraphedStudent(
            AutocastStudent(student.converted("cueq")), types, n, capacity)
    for name, build in builders.items():
        try:
            runner = build()
            runner.set_edges(i, j, sh)
            _, f_var = runner(xw)
            row[f"{name}_max_abs_force_diff"] = float((f_var - f_ref).abs().max())
            row[f"{name}_ms"] = time_call_ms(lambda r=runner: r(xw), repeats=50)
            del runner
        except Exception as exc:  # report and keep going; eager still stands
            row[f"{name}_error"] = error_text(exc)
        torch.cuda.empty_cache()

    try:
        half = load_student(path, device).to(torch.bfloat16)
        runner = CompiledGraphedStudent(half, types, n, capacity, dtype=torch.bfloat16)
        runner.set_edges(i, j, sh)
        _, f_half = runner(xw)
        row["compiled_graph_bf16_max_abs_force_diff"] = float((f_half.float() - f_ref).abs().max())
        row["compiled_graph_bf16_ms"] = time_call_ms(lambda r=runner: r(xw), repeats=50)
    except Exception as exc:
        row["compiled_graph_bf16_error"] = error_text(exc)
    torch.cuda.empty_cache()

    if student.config.get("kind") == "mace":
        for kind in (("e3nn", "cueq", "oeq") if backend == "e3nn" else (backend,)):
            try:
                row[f"train_update_ms_{kind}"] = time_train_update(path, kind, xw, i, j, sh,
                                                                   numbers, device)
            except Exception as exc:
                row[f"train_update_error_{kind}"] = error_text(exc)
            torch.cuda.empty_cache()
    return row


def best_force_call_ms(row: dict) -> float:
    """The fastest call whose forces match eager e3nn within FORCE_TOL."""
    times = [row["eager_ms"]]
    for key, value in row.items():
        if key.endswith("_ms") and key != "eager_ms":
            diff = row.get(f"{key[:-3]}_max_abs_force_diff")
            if diff is not None and diff < FORCE_TOL:
                times.append(value)
    return min(times)


def same_edges(a, b) -> bool:
    """Whether two (x, i, j, shifts, ...) lists hold the same edges with the same vectors."""
    def rows(x, i, j, s):
        r = (x[j] - x[i] + s).double().cpu().numpy()
        key = np.column_stack((i.cpu().numpy(), j.cpu().numpy(), np.round(r, 3)))
        return key[np.lexsort(key.T[::-1])]

    ra, rb = rows(*a[:4]), rows(*b[:4])
    return ra.shape == rb.shape and bool(np.allclose(ra, rb, atol=2e-3))


def time_neighbours(atoms, cell, device: str, radii: dict) -> dict:
    """Full-list builds from unwrapped positions: vesin (CPU, with the copy to the GPU) and ALCHEMI."""
    from frost.neighbours import edges_for, edges_on_gpu
    from frost.teacher import time_call_ms

    x_np = atoms.get_positions()
    x = torch.as_tensor(x_np, dtype=torch.float64, device=device)
    out = {}
    for name, radius in radii.items():
        def vesin(r=radius):
            return edges_for(x_np, cell, r, device, torch.float64, wrap_positions=False)

        def alchemi(r=radius):
            return edges_on_gpu(x, cell, r, wrap_positions=False)

        ref = vesin()
        out[f"{name}_edges"] = int(ref[1].numel())
        out[f"{name}_vesin_ms"] = time_call_ms(vesin, repeats=10, warmup=2)
        try:
            out[f"{name}_alchemi_same_edges"] = same_edges(ref, alchemi())
            out[f"{name}_alchemi_ms"] = time_call_ms(alchemi, repeats=10, warmup=2)
        except Exception as exc:
            out[f"{name}_alchemi_error"] = error_text(exc)
    return out


def main() -> int:
    import h5py

    from frost.teacher import frame_atoms

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="silica1152_T300")
    ap.add_argument("--frame", type=int, default=4000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="", help="suffix for the result file")
    ap.add_argument("--only", default="", help="comma-separated substrings of student names to time")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    with h5py.File(TRAJ / f"{args.run}.h5", "r") as fh:
        numbers = np.asarray(fh.attrs["numbers"], dtype=np.int64)
        cell = np.asarray(fh.attrs["cell"], dtype=np.float64)
        positions = np.asarray(fh["positions"][args.frame], dtype=np.float64)
    atoms = frame_atoms(numbers, positions, cell)

    report: dict = {"run": args.run, "frame": args.frame, "n_atoms": len(atoms)}
    report["neighbours"] = time_neighbours(atoms, cell, args.device,
                                           {"teacher_7.0A": 7.0, "student_4.3A": 4.3,
                                            "student_5.3A": 5.3})
    print("neighbours:", report["neighbours"], flush=True)
    report["teacher"] = time_teachers(atoms, cell, args.device)
    print("teacher:", report["teacher"], flush=True)
    t = report["teacher"]
    teacher_ms = t.get("teacher_cueq_graph_force_ms", t.get("teacher_cueq_force_ms",
                                                            t.get("teacher_e3nn_force_ms")))
    report["c_S_denominator_ms"] = teacher_ms
    report["students"] = {}
    for path in sorted(STUDENTS.glob(f"{args.run}*.pt")):
        if args.only and not any(s in path.stem for s in args.only.split(",")):
            continue
        row = time_variants(path, atoms, cell, numbers, args.device)
        row["best_force_call_ms"] = best_force_call_ms(row)
        row["c_S_best_vs_fastest_teacher"] = row["best_force_call_ms"] / teacher_ms if teacher_ms else None
        report["students"][path.stem] = row
        times = " | ".join(f"{k[:-3]} {v:.2f}" for k, v in row.items() if k.endswith("_ms"))
        print(f"{path.stem}: {times} ms | teacher {teacher_ms:.1f} ms | c_S best "
              f"{row['c_S_best_vs_fastest_teacher']:.3f}", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out = RESULTS / f"student_cost_{args.run}{suffix}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
