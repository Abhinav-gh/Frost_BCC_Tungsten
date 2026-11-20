"""E5: train the in-situ student on a stream of teacher calls, then measure it as E4 did.

Week 1 of the reframed plan (plan.md section 5.A, note section 12). E4 measured
rate separation with MACE-MP-0 small, a generic model that was never trained for
the role and costs a third of the teacher. This replays what an in-situ run would
see and answers two gate questions directly:

* G1, absolute part: does a purpose-built student bring the mean staleness over
  a 4 fs outer step under 20 meV/A on 300 K silica?
* G2: does it cost at most 0.05 of a teacher force call, against the
  cuEquivariance teacher as well as e3nn?

Protocol, per E1 run. Every statistic covers every atom of every frame, never a
single atom or edge.
  1. Teacher calls: MACE-MP-0 medium on every 4th frame of the first 2000 steps,
     the 500 labels a k = 4 schedule would produce over 2 ps.
  2. Train the student on that stream with the force, force-change and
     energy-change loss (frost.insitu). Two student families:
       frost  frost.student.FrostStudent, strictly local, body order 3
       mace   frost.student_mace.MaceStudent, a small MACE trained from scratch
  3. Evaluate on three later windows it never saw (steps 4000, 7000 and 9000;
     33 frames each) with the E4 statistics, for the student and for MACE-MP-0
     small on the same frames.
  4. Time teacher (e3nn and cuEquivariance) and student (eager and CUDA graph)
     force calls on the same frame and GPU.

Output: data/results/e5/student_replay[_<tag>].json; weights in
data/models/students/<run>[_<tag>].pt.

Usage (guarded launcher; GPU, float32)
  scripts/safe_run.sh e5_replay .venvs/frost-mace/bin/python \\
      experiments/e5_student/replay.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TRAJ = ROOT / "data" / "trajectories"
RESULTS = ROOT / "data" / "results" / "e5"
STUDENTS = ROOT / "data" / "models" / "students"

RUNS = (
    "si512_T300", "silica1152_T300", "silica_glass1152_T300", "water216_T300",
    "cu500_T300", "li3po4_T300", "si512_T1000", "silica1152_T1000",
)
KS = (1, 2, 4, 8, 16, 32)


def residual_stats(f_t, f_s, e_t, e_s, temperature: float, n_atoms: int, ks) -> dict:
    from frost.staleness import drift, energy_stats, hold, lagged, linear, rms

    r = f_t - f_s
    er = e_t - e_s
    return {
        "level": rms(r),
        "drift_R": {k: drift(r, k) for k in ks},
        "hold_R": {k: hold(r, k) for k in ks},
        "lin_R": {k: linear(r, k) for k in ks},
        "lag1_R": {k: lagged(r, k) for k in ks},
        "energy": {k: energy_stats(er, k, temperature, n_atoms) for k in ks},
    }


def make_student(args, species, mean_neighbours: float):
    if args.student == "mace":
        from frost.student_mace import MaceStudent

        return MaceStudent(species, cutoff=args.cutoff, channels=args.channels,
                           max_ell=args.max_ell, layers=args.layers,
                           correlation=args.correlation, mean_neighbours=mean_neighbours,
                           backend=args.student_backend, num_bessel=args.num_bessel)
    if args.student == "nc":  # force-only, DMTS-NC's contract on our environment features (frost.student_nc)
        from frost.student_nc import ForceStudent

        student = ForceStudent(species, cutoff=args.cutoff, channels=args.channels, mix=args.mix,
                               hidden=args.hidden, lmax=args.lmax, body4=args.body4)
        student.set_neighbour_normalisation(mean_neighbours)
        return student
    from frost.student import FrostStudent

    student = FrostStudent(species, cutoff=args.cutoff, channels=args.channels, mix=args.mix,
                           hidden=args.hidden, lmax=args.lmax, body4=args.body4)
    student.set_neighbour_normalisation(mean_neighbours)
    return student


def time_student(student, types, n_atoms: int, x, i, j, shifts) -> dict:
    """Eager and CUDA-graph force-call times for the student on one frame."""
    from frost.student import GraphedStudent, energy_and_forces
    from frost.teacher import time_call_ms

    def eager():
        return energy_and_forces(student, x, types, i, j, shifts)

    out = {"student_eager_ms": time_call_ms(eager, repeats=20)}
    try:
        graphed = GraphedStudent(student, types, n_atoms, capacity=int(1.2 * i.numel()) + 16)
        graphed.set_edges(i, j, shifts)
        out["student_graph_ms"] = time_call_ms(lambda: graphed(x), repeats=50)
        _, f_graph = graphed(x)
        _, f_eager = eager()
        out["graph_vs_eager_max_abs_eV_per_A"] = float((f_graph - f_eager).abs().max())
    except Exception as exc:  # the eager number still stands
        out["student_graph_error"] = f"{type(exc).__name__}: {exc}"[:200]
    return out


def time_teachers(atoms, device: str, teacher: str = "mp-medium") -> dict:
    """The teacher's force and energy-only calls with e3nn and with cuEquivariance kernels."""
    from frost.teacher import mace_forces, time_mace_ms
    from frost.teachers import load_calc

    out = {}
    for tag, cueq in (("e3nn", False), ("cueq", True)):
        try:
            calc = load_calc(teacher, device, "float32", enable_cueq=cueq)
            _, _, batch = mace_forces(calc, atoms, device)
            out[f"teacher_{tag}_force_ms"] = time_mace_ms(calc.models[0], batch)
            out[f"teacher_{tag}_energy_ms"] = time_mace_ms(calc.models[0], batch, force=False)
            del calc, batch
        except Exception as exc:  # e3nn always works; record why cuEq did not
            out[f"teacher_{tag}_error"] = f"{type(exc).__name__}: {exc}"[:200]
        torch.cuda.empty_cache()
    return out


def main() -> int:
    from mace.calculators import mace_mp

    from frost.insitu import Frame, InSituTrainer, LabelBuffer
    from frost.neighbours import edges_for
    from frost.staleness import drift, rms
    from frost.student import energy_and_forces
    from frost.teacher import frame_atoms, mace_forces
    from frost.trajectory import TrajectoryReader

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--only", default="", help="comma-separated subset of runs")
    ap.add_argument("--tag", default="", help="suffix for the result and weight files")
    ap.add_argument("--student", choices=("frost", "mace", "nc"), default="frost")
    ap.add_argument("--cutoff", type=float, default=4.0)
    ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--mix", type=int, default=0, help="frost: mixing width (0 = channels)")
    ap.add_argument("--hidden", type=int, default=64, help="frost: readout width")
    ap.add_argument("--lmax", type=int, default=2, help="frost: 2 or 3")
    ap.add_argument("--body4", action="store_true",
                    help="frost: add NEP4's four-body invariants (a^T M b and tr(A B C) of l = 1, 2 moments)")
    ap.add_argument("--max-ell", type=int, default=2, help="mace: spherical harmonics order")
    ap.add_argument("--layers", type=int, default=2, help="mace: interactions")
    ap.add_argument("--correlation", type=int, default=3, help="mace: body order - 1")
    ap.add_argument("--num-bessel", type=int, default=8,
                    help="mace: radial basis functions; more resolves stiff bonds")
    ap.add_argument("--student-backend", choices=("e3nn", "cueq", "oeq"), default="e3nn",
                    help="mace: kernels for training and timing the student")
    ap.add_argument("--teacher-cueq", action="store_true",
                    help="label with cuEquivariance teacher kernels (same model, faster)")
    ap.add_argument("--label-stride", type=int, default=4)
    ap.add_argument("--train-steps", type=int, default=2000,
                    help="trajectory steps whose teacher calls form the training stream")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=4, help="pairs of consecutive calls per update")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--w-dforce", type=float, default=1.0)
    ap.add_argument("--w-denergy", type=float, default=10.0)
    ap.add_argument("--max-gap", type=int, default=0,
                    help="steps (fs) between the two frames of an energy-change pair "
                         "(0: --label-stride, i.e. consecutive calls). The statistics contract's "
                         "acceptance depends on how the energy residual drifts over a whole draft, "
                         "so a student for it should see pairs a draft apart (plan §1.14)")
    ap.add_argument("--spread-labels", action="store_true",
                    help="draw pairs from any two calls up to --max-gap apart, not only that gap")
    ap.add_argument("--ring-stride", type=int, default=0,
                    help="bead dumps: frames per bead block (dump fs / dt); pairs this many frames apart are ring "
                         "partners at one time, and --ring-batch of them join each update's change terms")
    ap.add_argument("--ring-batch", type=int, default=4)
    ap.add_argument("--species-balance", action="store_true",
                    help="every element's force error counts equally in the loss (a solute among many solvent atoms)")
    ap.add_argument("--starts", default="4000,7000,9000")
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--teacher-ext", default="",
                    help="label with a teacher served from its own environment (frost.external_teacher), e.g. "
                         "pet:pet-omatpes-v2 or uma:uma-m-1p1:omat, instead of the trajectory's MACE teacher")
    args = ap.parse_args()
    args.mix = args.mix or args.channels
    torch.set_num_threads(args.threads)
    dev = args.device
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = RESULTS / f"student_replay{suffix}.json"

    from frost.teachers import load_calc, teacher_of

    # each run is labelled by the teacher its trajectory file names (frost.teachers)
    teachers = {"mp-medium": mace_mp(model="medium", device=dev, default_dtype="float32",
                                     enable_cueq=args.teacher_cueq)}
    small_mp = mace_mp(model="small", device=dev, default_dtype="float32")
    # --only may name any trajectory in data/trajectories (e.g. the liquid cu500_T1600);
    # without it the default E1 list runs as before
    runs = args.only.split(",") if args.only else list(RUNS)
    starts = [int(s) for s in args.starts.split(",")]
    ks = [k for k in KS if k <= args.window]
    report = json.loads(out_path.read_text()) if out_path.exists() else {"runs": {}}

    for run in runs:
        t0 = time.perf_counter()
        path = TRAJ / f"{run}.h5"
        if not path.exists():
            print(f"{run}: missing trajectory, skipped", flush=True)
            continue
        temperature = float(run.split("_T")[1])
        with h5py.File(path, "r") as fh:
            teacher_name = teacher_of(fh.attrs)
        from frost.teachers import TEACHERS

        if teacher_name not in TEACHERS:
            # a trajectory made by a model outside frost.teachers (e.g. the MTS-Allegro rebuild's water): its labels must
            # come from --teacher-ext, and MACE-MP-0 medium stands in only for the side columns (small model, timing)
            if not args.teacher_ext:
                raise SystemExit(f"{run}: trajectory teacher {teacher_name!r} is not a MACE teacher; pass --teacher-ext")
            print(f"{run}: trajectory teacher {teacher_name!r}; side columns use mp-medium", flush=True)
            teacher_name = "mp-medium"
        if teacher_name not in teachers:
            teachers[teacher_name] = load_calc(teacher_name, dev, "float32", enable_cueq=args.teacher_cueq)
        teacher = teachers[teacher_name]
        # the cheaper foundation model of the same family, for the "small" comparison column
        small = small_mp if teacher_name == "mp-medium" else teachers.setdefault(
            "off-small", load_calc("off-small", dev, "float32", enable_cueq=False))
        label_steps = list(range(0, args.train_steps, args.label_stride))
        with TrajectoryReader(path) as reader:
            numbers = reader.numbers.copy()
            cell = reader.cell.copy()
            label_pos = [np.asarray(reader.positions[t]) for t in label_steps]
            eval_pos = [
                [np.asarray(reader.positions[s + j]) for j in range(args.window + 1)]
                for s in starts
            ]
        n_atoms = int(numbers.size)

        if args.teacher_ext:
            from frost.external_teacher import ExternalTeacher

            ext = ExternalTeacher(args.teacher_ext, numbers, cell, dev)

            def label(atoms):
                e, f = ext.numpy(atoms.get_positions())
                return f, e
        else:
            def label(atoms):
                f, e, _ = mace_forces(teacher, atoms, dev)
                return f, e

        # 1. the teacher calls a k = 4 schedule would make
        buffer = LabelBuffer(capacity=len(label_steps))
        edges = 0
        for step, pos in zip(label_steps, label_pos, strict=True):
            atoms = frame_atoms(numbers, pos, cell)
            f_t, e_t = label(atoms)
            xw, ei, ej, sh, _ = edges_for(atoms.get_positions(), cell, args.cutoff, dev)
            edges += int(ei.numel())
            forces = torch.as_tensor(f_t, dtype=torch.float32, device=dev)
            buffer.add(Frame(step, xw, forces, e_t, ei, ej, sh))
        mean_neighbours = edges / (len(label_steps) * n_atoms)
        t_label = time.perf_counter() - t0

        torch.manual_seed(args.seed)
        student = make_student(args, np.unique(numbers), mean_neighbours).to(dev)
        types = student.types(torch.as_tensor(numbers, device=dev))

        # 2. train on that stream
        trainer = InSituTrainer(student, types, lr=args.lr, w_dforce=args.w_dforce,
                                w_denergy=args.w_denergy,
                                max_gap=args.max_gap or args.label_stride,
                                spread=args.spread_labels, seed=args.seed,
                                ring_stride=args.ring_stride, ring_batch=args.ring_batch if args.ring_stride else 0,
                                species_balance=args.species_balance)
        log = []
        for it in range(args.iters):
            trainer.set_lr(args.lr * (0.05 + 0.475 * (1.0 + math.cos(math.pi * it / args.iters))))
            due = it % 500 == 0 or it == args.iters - 1
            stats = trainer.step(buffer, batch_pairs=args.batch, report=due)
            if due and stats:
                log.append({"iter": it, **stats})
        t_train = time.perf_counter() - t0 - t_label
        del buffer
        torch.cuda.empty_cache()

        # 3. unseen windows: teacher, student, and MACE-MP-0 small on the same frames
        shape = (len(starts), args.window + 1)
        f_teacher, e_teacher = np.zeros((*shape, n_atoms, 3)), np.zeros(shape)
        f_student, e_student = np.zeros((*shape, n_atoms, 3)), np.zeros(shape)
        f_small, e_small = np.zeros((*shape, n_atoms, 3)), np.zeros(shape)
        for w, window in enumerate(eval_pos):
            for j, pos in enumerate(window):
                atoms = frame_atoms(numbers, pos, cell)
                f_teacher[w, j], e_teacher[w, j] = label(atoms)
                f_small[w, j], e_small[w, j], _ = mace_forces(small, atoms, dev)
                xw, ei, ej, sh, _ = edges_for(atoms.get_positions(), cell, args.cutoff, dev)
                e_s, f_s = energy_and_forces(student, xw, types, ei, ej, sh)
                f_student[w, j] = f_s.detach().double().cpu().numpy()
                e_student[w, j] = float(e_s.detach().double().sum())

        entry = {
            "n_atoms": n_atoms,
            "temperature_K": temperature,
            "config": {k: v for k, v in vars(args).items() if k not in ("only", "device", "threads")},
            "student": student.config,
            "n_params": int(sum(p.numel() for p in student.parameters())),
            "student_edges_per_atom": mean_neighbours,
            "teacher_calls": len(label_steps),
            "label_seconds": t_label,
            "train_seconds": t_train,
            "train_log": log,
            "teacher_force_rms": rms(f_teacher),
            "drift_F_teacher": {k: drift(f_teacher, k) for k in ks},
            "frost_student": residual_stats(f_teacher, f_student, e_teacher, e_student,
                                            temperature, n_atoms, ks),
            "mp0_small": residual_stats(f_teacher, f_small, e_teacher, e_small,
                                        temperature, n_atoms, ks),
        }

        # 4. cost on one frame, same GPU
        atoms = frame_atoms(numbers, eval_pos[0][0], cell)
        timing = time_teachers(atoms, dev, teacher_name)
        xw, ei, ej, sh, _ = edges_for(atoms.get_positions(), cell, args.cutoff, dev)
        timing.update(time_student(student, types, n_atoms, xw, ei, ej, sh))
        entry["timing"] = timing
        best = "student_graph_ms" if "student_graph_ms" in timing else "student_eager_ms"
        for tag in ("e3nn", "cueq"):
            if f"teacher_{tag}_force_ms" in timing:
                entry[f"c_S_vs_{tag}"] = timing[best] / timing[f"teacher_{tag}_force_ms"]

        STUDENTS.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": student.state_dict(), "config": student.config},
                   STUDENTS / f"{run}{suffix}.pt")
        report["runs"][run] = entry
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))

        fs, sm = entry["frost_student"], entry["mp0_small"]
        mev = 1e3
        k_acc = 16 if 16 in ks else ks[-1]
        print(
            f"{run}{suffix}: h_R(4) student {fs['hold_R'][4] * mev:.1f} vs small "
            f"{sm['hold_R'][4] * mev:.1f} meV/A | level {fs['level'] * mev:.0f} vs "
            f"{sm['level'] * mev:.0f} | acc({k_acc}) {fs['energy'][k_acc]['acceptance_proxy']:.2f}"
            f" vs {sm['energy'][k_acc]['acceptance_proxy']:.2f} | c_S vs e3nn "
            f"{entry.get('c_S_vs_e3nn', float('nan')):.3f} vs cueq "
            f"{entry.get('c_S_vs_cueq', float('nan')):.3f} [{time.perf_counter() - t0:.0f}s]",
            flush=True,
        )

    print(f"-> {out_path}")
    return 0 if report["runs"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
