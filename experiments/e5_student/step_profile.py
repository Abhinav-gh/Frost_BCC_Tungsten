"""Where the time of one multirate inner step goes, on one frame and GPU.

The online loop's student clock holds more than the student's force call. It
also holds:
- the integrator's kicks, drifts and Langevin noise;
- the neighbour-list check and its host-device synchronisation;
- list rebuilds.

This times each piece alone and the whole step, with the settings the online
loop uses, so the overhead is attributed rather than guessed.

Pieces (median ms, CUDA-synchronised):
- student_call: the CUDA-graph student force call;
- list_check: the Verlet displacement check when no rebuild is due;
- list_rebuild_<builder>: a forced rebuild plus set_edges, with vesin and with ALCHEMI;
- integrator: one BAOAB inner step with a fixed force (no model);
- inner_step: one BAOAB inner step with the student, exactly as online.py runs
  it (its rebuild count is reported);
- teacher_<eager|graph>: the cuEquivariance teacher force call on its 1 A skin list.

Output: data/results/e5/step_profile_<run>[_<tag>].json

Usage (guarded launcher; GPU)
  scripts/safe_run.sh e5_step_profile .venvs/frost-mace/bin/python \\
      experiments/e5_student/step_profile.py --run silica1152_T300 --student-backend cueq
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


def main() -> int:
    import h5py
    from mace.calculators import mace_mp

    from frost.integrators import Integrator, Langevin, MDState
    from frost.neighbours import BUILDERS, VerletList
    from frost.student import GraphedStudent
    from frost.student_mace import BACKENDS
    from frost.students import load_student
    from frost.teacher import MaceTeacher, frame_atoms, time_call_ms
    from frost.teacher_fast import GraphedTeacher

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="silica1152_T300")
    ap.add_argument("--frame", type=int, default=9000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--student", default="", help="saved student (default: <run>_m4c16.pt)")
    ap.add_argument("--student-backend", choices=BACKENDS, default="e3nn")
    ap.add_argument("--student-skin", type=float, default=0.3)
    ap.add_argument("--neighbours", choices=BUILDERS, default="alchemi")
    ap.add_argument("--compile", default="", help="Inductor mode for a descriptor student, as md_runtime --compile")
    ap.add_argument("--edge-forces", action="store_true", help="the teacher's edge-force output, as the runtime runs it")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dev = args.device
    temperature = float(args.run.split("_T")[1])

    with h5py.File(TRAJ / f"{args.run}.h5", "r") as fh:
        numbers = np.asarray(fh.attrs["numbers"], dtype=np.int64)
        cell = np.asarray(fh.attrs["cell"], dtype=np.float64)
        positions = np.asarray(fh["positions"][args.frame], dtype=np.float64)
        velocities = np.asarray(fh["velocities"][args.frame], dtype=np.float64)
    atoms = frame_atoms(numbers, positions, cell)
    n = len(atoms)
    x = torch.as_tensor(atoms.get_positions(), dtype=torch.float64, device=dev)
    v0 = torch.as_tensor(velocities, dtype=torch.float64, device=dev)
    masses = torch.as_tensor(atoms.get_masses(), dtype=torch.float64, device=dev).unsqueeze(1)

    path = Path(args.student) if args.student else STUDENTS / f"{args.run}_m4c16.pt"
    student = load_student(path, dev)
    if args.student_backend != student.config.get("backend", "e3nn"):
        student = student.converted(args.student_backend)
    types = student.types(torch.as_tensor(numbers, device=dev))
    cutoff = float(student.config["cutoff"])
    s_list = VerletList(cell, cutoff, args.student_skin, dev, wrap_positions=False,
                        builder=args.neighbours)
    s_list.update(x, force=True)
    graphed = GraphedStudent(student, types, n, capacity=int(1.3 * s_list.i.numel()) + 64)
    graphed.set_edges(s_list.i, s_list.j, s_list.shifts)
    if args.compile:  # exactly as runtime.md.build compiles a descriptor student
        from frost.student_fast import CompiledGraphedStudent

        graphed = CompiledGraphedStudent(student, types, n, capacity=int(1.3 * s_list.i.numel()) + 64, mode=args.compile)
        graphed.set_edges(s_list.i, s_list.j, s_list.shifts)

    out: dict = {"run": args.run, "n_atoms": n, "student": student.config,
                 "student_edges": int(s_list.i.numel()), "neighbours": args.neighbours}
    out["student_call_ms"] = time_call_ms(lambda: graphed(x), repeats=100)
    out["list_check_ms"] = time_call_ms(lambda: s_list.update(x), repeats=100)
    for builder in BUILDERS:
        verlet = VerletList(cell, cutoff, args.student_skin, dev, wrap_positions=False,
                            builder=builder)

        def rebuild(lst=verlet):
            lst.update(x, force=True)
            graphed.set_edges(lst.i, lst.j, lst.shifts)

        out[f"list_rebuild_{builder}_ms"] = time_call_ms(rebuild, repeats=20)
    graphed.set_edges(s_list.i, s_list.j, s_list.shifts)

    e0, f0 = graphed(x)
    e0, f0 = e0.detach().sum(), f0.detach().double().clone()
    fixed_state = MDState(x.clone(), v0.clone(), masses)
    fixed = Integrator(fixed_state, 1.0, Langevin(1.0, 0.01, temperature, masses, 0))
    out["integrator_ms"] = time_call_ms(lambda: fixed.inner_step(lambda p: (e0, f0, p), f0),
                                        repeats=100)

    def student_force(p):
        if s_list.update(p)[1]:
            graphed.set_edges(s_list.i, s_list.j, s_list.shifts)
        e, f = graphed(p)
        return e.detach().sum(), f.detach().double(), p

    state = MDState(x.clone(), v0.clone(), masses)
    integ = Integrator(state, 1.0, Langevin(1.0, 0.01, temperature, masses, 1))
    holder = {"f": student_force(state.x)[1]}
    rebuilds_before = s_list.rebuilds

    def inner():
        _, holder["f"] = integ.inner_step(student_force, holder["f"])

    out["inner_step_ms"] = time_call_ms(inner, repeats=200, warmup=20)
    out["inner_step_rebuilds_in_220_steps"] = s_list.rebuilds - rebuilds_before

    calc = mace_mp(model="medium", device=dev, default_dtype="float32", enable_cueq=True)
    for name, cls in (("eager", MaceTeacher), ("graph", GraphedTeacher)):
        teacher = cls(calc, atoms, dev, edge_forces=True) if (args.edge_forces and cls is GraphedTeacher) else \
            cls(calc, atoms, dev)
        t_list = VerletList(cell, teacher.r_max, 1.0, dev, wrap_positions=False,
                            builder=args.neighbours)
        t_list.update(x, force=True)
        teacher.set_edges(t_list.i, t_list.j, t_list.shifts, t_list.unit_shifts)
        out[f"teacher_{name}_ms"] = time_call_ms(lambda t=teacher: t(x), repeats=20)
        del teacher
        torch.cuda.empty_cache()

    out["overhead_ms_per_inner_step"] = out["inner_step_ms"] - out["student_call_ms"]
    RESULTS.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    target = RESULTS / f"step_profile_{args.run}{suffix}.json"
    target.write_text(json.dumps(out, indent=2))
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in out.items() if k != "student"},
          f"-> {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
