"""E5b: the multirate loop on live dynamics, with the student trained while it runs.

Replay (E5a) scores a student offline. This runs the loop end to end from an E1
frame, in three phases.

  baseline   plain teacher MD (BAOAB Langevin, 1 fs, E1 friction) on the same
             kernels and neighbour lists: the wall time per fs Frost must beat.
             It is timed twice on the same trajectory, before and after the
             other phases, and the faster window is the denominator of every
             speedup.
  bootstrap  the same teacher dynamics while the student trains on every
             teacher call. Before each update the student is scored on the new
             call, so the realised staleness is logged from the first step.
  multirate  BAOAB-RESPA with the student every inner step and the teacher every
             k steps. The controller sets k from the realised staleness against
             the budget epsilon. Training continues on the calls the schedule
             makes until --freeze-after-fs, then the student is frozen.

After each training update the residual is recomputed against the new student
from the same teacher call, so F_S + R = F_T holds at every outer-step boundary.
Once the student is frozen that extra student call is skipped.

Neighbour lists. Teacher and student each keep a Verlet list built from unwrapped
positions, so each rebuilds on its own schedule: the teacher's on --skin, the
student's on the tighter --student-skin. --neighbours picks vesin on the CPU or
NVIDIA ALCHEMI's cell list on the GPU. The teacher runs on its list only after
its forces are checked against MACE's own neighbour list on every atom of the
starting frame.

Kernels. --teacher-cueq gives the teacher cuEquivariance kernels, and
--teacher-graph replays its force call from a CUDA graph. Both apply to the
baseline exactly as to the multirate path. --student-backend runs a MACE student
on e3nn, cuEquivariance or OpenEquivariance kernels, always inside a CUDA graph.

Output: data/results/e5/online_<run>[_<tag>].json

Usage (guarded launcher; GPU)
  scripts/safe_run.sh e5_online_si .venvs/frost-mace/bin/python \\
      experiments/e5_student/online.py --run si512_T300 --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TRAJ = ROOT / "data" / "trajectories"
OUT_DIR = ROOT / "data" / "results" / "e5"


def rms_mev(t: torch.Tensor, masses: torch.Tensor | None = None) -> float:
    """RMS over atoms of a residual change, per unit mass when masses are given.

    Integration error follows acceleration, so the certificate divides by mass:
    meV/(A amu) rather than meV/A (plan section 1.11).
    """
    if masses is not None:
        t = t / masses
    return float(t.norm(dim=1).pow(2).mean().sqrt()) * 1e3


def main() -> int:
    import h5py
    from mace.calculators import mace_mp

    from frost import certified
    from frost.controller import DriftMonitor, IntervalController
    from frost.insitu import Frame, InSituTrainer, LabelBuffer
    from frost.integrators import Integrator, Langevin, MDState
    from frost.neighbours import BUILDERS, VerletList
    from frost.student import GraphedStudent
    from frost.student_mace import BACKENDS
    from frost.students import load_student, make_student
    from frost.teacher import MaceTeacher, frame_atoms, mace_forces
    from frost.teacher_fast import GraphedTeacher

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="si512_T300")
    ap.add_argument("--tag", default="", help="suffix for the result file")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--start", type=int, default=9000, help="E1 frame to start from")
    ap.add_argument("--fs", type=int, default=2000, help="simulated fs, bootstrap included")
    ap.add_argument("--bootstrap-fs", type=int, default=300)
    ap.add_argument("--bootstrap-exit-k", type=int, default=0,
                    help="leave the bootstrap early once the one-step staleness times this k is "
                         "within half the budget (0: always run --bootstrap-fs)")
    ap.add_argument("--bootstrap-min-fs", type=int, default=50)
    ap.add_argument("--freeze-after-fs", type=int, default=1000)
    ap.add_argument("--freeze-when-quiet", type=int, default=0,
                    help="freeze early after this many consecutive outer steps at k_max within "
                         "half the budget (0: off)")
    ap.add_argument("--train-budget-s", type=float, default=0.0,
                    help="freeze once training has used this many wall-clock seconds (0: no cap)")
    ap.add_argument("--baseline-fs", type=int, default=300)
    ap.add_argument("--units", choices=("acceleration", "force"), default="acceleration",
                    help="certificate units: meV/(A amu), or meV/A as in weeks 1-2")
    ap.add_argument("--epsilon", type=float, default=None,
                    help="staleness budget; default 2.0 for acceleration, 20.0 for force")
    ap.add_argument("--fastest-period-fs", type=float, default=0.0,
                    help="fastest vibrational period; caps the outer step at 0.4 of it "
                         "(0: no resonance cap). Silica 30, Li3PO4 33, water 9")
    ap.add_argument("--k-max", type=int, default=16)
    ap.add_argument("--skin", type=float, default=1.0, help="teacher Verlet skin, A")
    ap.add_argument("--student-skin", type=float, default=0.3, help="student Verlet skin, A")
    ap.add_argument("--neighbours", choices=BUILDERS, default="vesin", help="neighbour-list builder")
    ap.add_argument("--friction", type=float, default=0.01, help="per fs, as in E1")
    ap.add_argument("--teacher-cueq", action="store_true", help="cuEquivariance teacher kernels")
    ap.add_argument("--teacher-graph", action="store_true",
                    help="replay the teacher force call from a CUDA graph (baseline included)")
    ap.add_argument("--student", choices=("frost", "mace"), default="frost")
    ap.add_argument("--student-backend", choices=BACKENDS, default="e3nn",
                    help="kernels of a MACE student")
    ap.add_argument("--cutoff", type=float, default=4.0)
    ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--mix", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lmax", type=int, default=2)
    ap.add_argument("--max-ell", type=int, default=2)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--correlation", type=int, default=3)
    ap.add_argument("--train-per-call", type=int, default=2)
    ap.add_argument("--spread-pairs", action="store_true",
                    help="force-change labels from any two calls up to k_max steps apart")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant",
                    help="cosine: from --lr at t=0 to 5%% of it at --freeze-after-fs, as replay trains")
    ap.add_argument("--buffer", type=int, default=2000)
    ap.add_argument("--fallback-fs", type=int, default=20)
    ap.add_argument("--warm-start", default="", help="student weights, e.g. from the replay run")
    ap.add_argument("--certified-store", default="data/results/e5/certified_intervals.json",
                    help="where intervals certified by earlier runs are kept (frost.certified). "
                         "The multirate phase starts at the ceiling a previous run held, so the "
                         "descent that discovers it is paid once per system and student")
    ap.add_argument("--no-warm-controller", action="store_true",
                    help="ignore the store and rediscover the interval from the cap")
    ap.add_argument("--ensemble", choices=("nvt", "nve"), default="nvt",
                    help="nvt keeps the BAOAB thermostat on both arms (the E1 setting). nve "
                         "removes it, which is what the drift clause needs: a thermostat absorbs "
                         "the energy a bad interval pumps in, so it hides the quantity the "
                         "monitor watches (plan section 1.14)")
    ap.add_argument("--drift-budget", type=float, default=None,
                    help="meV/atom projected to the horizon for the realised-drift clause; "
                         "default is whatever frost.controller.DriftMonitor ships (the G3 "
                         "criterion itself), so the two cannot drift apart")
    ap.add_argument("--drift-window-ps", type=float, default=10.0)
    ap.add_argument("--profile", action="store_true",
                    help="record the per-outer-step teacher/student/train breakdown. It "
                         "synchronises the device six times per outer step, inside the timed "
                         "window, so it lowers the measured speedup; leave it off for timing runs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    dev = args.device
    temperature = float(args.run.split("_T")[1])

    with h5py.File(TRAJ / f"{args.run}.h5", "r") as fh:
        numbers = np.asarray(fh.attrs["numbers"], dtype=np.int64)
        cell = np.asarray(fh.attrs["cell"], dtype=np.float64)
        positions = np.asarray(fh["positions"][args.start], dtype=np.float64)
        velocities = np.asarray(fh["velocities"][args.start], dtype=np.float64)
    atoms = frame_atoms(numbers, positions, cell)
    n = len(atoms)
    masses = torch.as_tensor(atoms.get_masses(), dtype=torch.float64, device=dev).unsqueeze(1)
    v0 = torch.as_tensor(velocities, dtype=torch.float64, device=dev)
    if args.epsilon is None:
        args.epsilon = 2.0 if args.units == "acceleration" else 20.0
    unit = "meV/(A amu)" if args.units == "acceleration" else "meV/A"
    scale = masses if args.units == "acceleration" else None

    teacher_calc = mace_mp(model="medium", device=dev, default_dtype="float32",
                           enable_cueq=args.teacher_cueq)
    teacher = (GraphedTeacher if args.teacher_graph else MaceTeacher)(teacher_calc, atoms, dev)
    t_list = VerletList(cell, teacher.r_max, args.skin, dev, wrap_positions=False,
                        builder=args.neighbours)

    x0 = torch.as_tensor(atoms.get_positions(), dtype=torch.float64, device=dev)
    t_list.update(x0, force=True)
    teacher.set_edges(t_list.i, t_list.j, t_list.shifts, t_list.unit_shifts)

    if args.warm_start:
        student = load_student(args.warm_start, dev)
        if args.student_backend != student.config.get("backend", "e3nn"):
            if student.config.get("kind") != "mace":
                raise SystemExit("--student-backend applies to MACE students only")
            student = student.converted(args.student_backend)
    else:
        probe = VerletList(cell, args.cutoff, 0.0, dev)
        probe.update(x0, force=True)
        student = make_student(args.student, np.unique(numbers), probe.i.numel() / n,
                               cutoff=args.cutoff, channels=args.channels, mix=args.mix or None,
                               hidden=args.hidden, lmax=args.lmax, max_ell=args.max_ell,
                               layers=args.layers, correlation=args.correlation,
                               backend=args.student_backend).to(dev)
    s_list = VerletList(cell, float(student.config["cutoff"]), args.student_skin, dev,
                        wrap_positions=False, builder=args.neighbours)
    s_list.update(x0, force=True)
    types = student.types(torch.as_tensor(numbers, device=dev))
    graphed = GraphedStudent(student, types, n, capacity=int(1.3 * s_list.i.numel()) + 64)
    graphed.set_edges(s_list.i, s_list.j, s_list.shifts)

    # the teacher on its Verlet list must reproduce MACE's own list on every atom
    f_skin = teacher(x0)[1].double().cpu().numpy()
    f_ref, _, _ = mace_forces(teacher_calc, atoms, dev)
    skin_error = float(np.abs(f_skin - f_ref).max())
    if skin_error > 1e-3:
        raise RuntimeError(f"teacher on the Verlet list differs by {skin_error:.2e} eV/A")

    def thermostat(seed):
        if args.ensemble == "nvt":
            return Langevin(1.0, args.friction, temperature, masses, seed)
        return None  # NVE: nothing absorbs the drift, so the monitor can see it

    monitor = DriftMonitor(window_ps=args.drift_window_ps)
    if args.drift_budget is not None:
        monitor.gain_budget = args.drift_budget
    drift_refusals = {"count": 0, "last": None, "peak": 0.0, "rungs": []}

    calls = {"teacher": 0, "student": 0, "teacher_rebuilds": 0, "student_rebuilds": 0}
    clock = {"teacher": 0.0, "student": 0.0, "train": 0.0}

    def now() -> float:
        torch.cuda.synchronize()
        return time.perf_counter()

    def mark() -> float:
        """``now()`` when profiling, a free clock read otherwise.

        The per-outer-step breakdown costs six device synchronisations inside the
        window it measures, while ``time_baseline`` below syncs exactly twice for a
        whole run. Charging that asymmetry to the multirate arm alone made it look
        3.69x -> 2.76x slower than the same loop measured by the fidelity harness
        (plan section 1.14), so the breakdown is off unless asked for.
        """
        return now() if args.profile else 0.0

    def teacher_force(x):
        if t_list.update(x)[1]:
            teacher.set_edges(t_list.i, t_list.j, t_list.shifts, t_list.unit_shifts)
            calls["teacher_rebuilds"] += 1
        e, f = teacher(x)
        calls["teacher"] += 1
        return e, f.double(), x

    def student_force(x):
        if s_list.update(x)[1]:
            graphed.set_edges(s_list.i, s_list.j, s_list.shifts)
            calls["student_rebuilds"] += 1
        e, f = graphed(x)
        calls["student"] += 1
        return e.detach().sum(), f.detach().double(), x

    def time_baseline():
        """Plain teacher MD from the starting frame; returns ms/fs, mean Epot/atom and T."""
        base = MDState(x0.clone(), v0.clone(), masses)
        integ_b = Integrator(base, 1.0, thermostat(args.seed))
        _, f, base.x = teacher_force(base.x)
        energies = []
        t0 = now()
        for _ in range(args.baseline_fs):
            e, f = integ_b.inner_step(teacher_force, f)
            energies.append(e)
        ms_per_fs = (now() - t0) / args.baseline_fs * 1e3
        return ms_per_fs, float(torch.stack(energies).mean()) / n, base.temperature()

    # ------------------------------------------------------------ baseline
    baseline_first, baseline_epot, baseline_T = time_baseline()
    calls_before = dict(calls)

    # ------------------------------------------------------------ bootstrap
    state = MDState(x0.clone(), v0.clone(), masses)
    integ = Integrator(state, 1.0, thermostat(args.seed + 1))
    controller = IntervalController(epsilon=args.epsilon, k=1, k_max=args.k_max,
                                    period_fs=args.fastest_period_fs or None, dt_fs=1.0)
    buffer = LabelBuffer(args.buffer)
    trainer = InSituTrainer(student, types, lr=args.lr, max_gap=args.k_max, seed=args.seed,
                            spread=args.spread_pairs)

    def label(step, x, f_t, e_t):
        buffer.add(Frame(step, x.detach().float().clone(), f_t.detach().float().clone(),
                         float(e_t), s_list.i, s_list.j, s_list.shifts.float()))

    # the student freezes at --freeze-after-fs, or earlier when it is already good
    # enough (quiet at k_max) or has used its training budget
    status = {"freeze_at": args.freeze_after_fs, "frozen_by": "schedule", "quiet": 0}

    def train(t_fs) -> bool:
        """Update the student on the label stream; False once it is frozen."""
        if t_fs >= status["freeze_at"] or args.train_per_call <= 0:
            return False
        if args.train_budget_s and clock["train"] >= args.train_budget_s:
            status["freeze_at"], status["frozen_by"] = t_fs, "training budget"
            return False
        if args.lr_schedule == "cosine":
            progress = min(t_fs / max(args.freeze_after_fs, 1), 1.0)
            trainer.set_lr(args.lr * (0.05 + 0.475 * (1.0 + math.cos(math.pi * progress))))
        for _ in range(args.train_per_call):
            trainer.step(buffer, batch_pairs=args.batch)
        return True

    log = []
    e_t, f_t, state.x = teacher_force(state.x)
    label(0, state.x, f_t, e_t)
    _, f_s, state.x = student_force(state.x)
    residual = f_t - f_s
    t_fs = 0
    t_boot = now()
    while t_fs < args.bootstrap_fs:
        a = now()
        e_t, f_t = integ.inner_step(teacher_force, f_t)
        b = now()
        _, f_s, state.x = student_force(state.x)
        staleness = rms_mev((f_t - f_s) - residual, scale)
        t_fs += 1
        label(t_fs, state.x, f_t, e_t)
        c = now()
        trained = train(t_fs)
        d = now()
        if trained:
            _, f_s, state.x = student_force(state.x)
        residual = f_t - f_s
        e = now()
        clock["teacher"] += b - a
        clock["student"] += (c - b) + (e - d)
        clock["train"] += d - c
        controller.observe(staleness, 1)
        if t_fs % 10 == 0:
            log.append({"t_fs": t_fs, "phase": "bootstrap", "k": 1, "staleness_meV": staleness,
                        "T": state.temperature()})
        # a student already projected to hold k = bootstrap_exit_k within half the
        # budget (linear drift) needs no more teacher-every-step training
        if (args.bootstrap_exit_k and t_fs >= args.bootstrap_min_fs
                and controller.ewma * args.bootstrap_exit_k <= 0.5 * args.epsilon):
            break
    bootstrap_s = now() - t_boot
    bootstrap_fs_used = t_fs

    # ------------------------------------------------------------ multirate
    # an interval a previous run held is evidence about this system with this
    # student under this certificate, so start there rather than growing to the
    # cap and descending: Si pays 60 ps of refusals to reach k = 10 otherwise
    warm = None
    if not args.no_warm_controller:
        warm = certified.lookup(args.certified_store, args.run, args.warm_start or "",
                                args.fastest_period_fs, args.epsilon, monitor.gain_budget)
    if warm is not None:
        # start_ceiling demotes a ceiling whose margin does not survive the
        # seed-to-seed scatter we measured on the glass, so one lucky run cannot
        # hand its luck to every later run of the system (frost.certified)
        controller.ceiling = warm.start_ceiling()
        controller.k = min(warm.k, controller.ceiling)
    k = max(2, controller.k)
    mr_fs = 0
    mr_epot = []
    fallback_fs = 0
    t_mr = now()
    while t_fs < args.fs:
        if controller.k <= 1:
            # never lose: plain teacher steps, then probe the multirate path again
            for _ in range(min(args.fallback_fs, args.fs - t_fs)):
                a = now()
                e_t, f_t = integ.inner_step(teacher_force, f_t)
                clock["teacher"] += now() - a
                t_fs += 1
                fallback_fs += 1
                label(t_fs, state.x, f_t, e_t)
            a = now()
            train(t_fs)
            b = now()
            _, f_s, state.x = student_force(state.x)
            residual = f_t - f_s
            clock["train"] += b - a
            clock["student"] += now() - b
            controller.k = k = 2
            continue
        k_use = min(k, args.fs - t_fs)
        training = t_fs < status["freeze_at"] and args.train_per_call > 0
        a = mark()
        _, f_s = integ.respa_outer(student_force, f_s, residual, k_use)
        b = mark()
        e_t, f_t, state.x = teacher_force(state.x)
        c = mark()
        r_new = f_t - f_s
        # this reduction syncs, but the controller needs the number: it is the
        # certificate's own cost, not instrumentation, so it stays in the timing
        staleness = rms_mev(r_new - residual, scale)
        integ.kick(r_new, 0.5 * k_use * integ.dt)
        t_fs += k_use
        mr_fs += k_use
        mr_epot.append(e_t)
        if training:
            label(t_fs, state.x, f_t, e_t)  # the buffer is never read once frozen
        if args.freeze_when_quiet and t_fs < status["freeze_at"]:
            quiet = k_use >= args.k_max and staleness / math.sqrt(3.0) < 0.5 * args.epsilon
            status["quiet"] = status["quiet"] + 1 if quiet else 0
            if status["quiet"] >= args.freeze_when_quiet:
                status["freeze_at"], status["frozen_by"] = t_fs, "quiet at k_max"
        # --train-budget-s reads clock["train"], so training is timed for real while
        # it runs; once frozen train() returns immediately and needs no sync
        d = now() if training else 0.0
        trained = train(t_fs)
        e = now() if training else 0.0
        if trained:
            # the student changed: re-anchor so F_S + R = F_T at this state
            _, f_s, state.x = student_force(state.x)
            residual = f_t - f_s
        else:
            residual = r_new
        g = mark()
        if training:
            clock["train"] += e - d
        if args.profile:
            clock["student"] += (b - a) + (g - e)
            clock["teacher"] += c - b
        log.append({"t_fs": t_fs, "phase": "multirate", "k": k_use, "staleness_meV": staleness,
                    "T": state.temperature() if args.profile else None,
                    "wall_s": (g if args.profile else time.perf_counter()) - t_mr})
        k = controller.observe(staleness, k_use)
        # the certificate's third clause: what this interval is realising, not what
        # its period predicts. Neither of the other two caught Si at k = 14. The
        # projection form needs min_ps of evidence, so it is slower to refuse than
        # the slope form it supersedes and never falsely refuses a decaying transient.
        total_meV = (float(e_t) + state.kinetic_energy()) / n * 1e3
        gain = monitor.projected_gain(total_meV, t_fs)
        if gain is not None:
            drift_refusals["peak"] = max(drift_refusals["peak"], gain)
        if gain is not None and gain > monitor.gain_budget and k_use > 2:
            # controller.refuse drops the interval now and ratchets a ceiling one
            # rung below it, so the growth rule can climb back toward the boundary
            # but never re-adopt a measured failure (frost.controller.refuse)
            k = controller.refuse(k_use)
            drift_refusals["count"] += 1
            drift_refusals["last"] = t_fs
            # which rungs were refused is what shows the ceiling is a boundary and
            # not just where a run happened to stop (frost-6e's request)
            drift_refusals["rungs"].append({"k": int(k_use), "t_fs": int(t_fs),
                                            "projection": round(float(gain), 4)})
            drift_refusals["peak"] = 0.0  # the peak belongs to the interval just left
            monitor.reset()  # what came before describes the interval just refused
    multirate_s = now() - t_mr
    calls_after = dict(calls)

    # ------------------------------------------------------------ baseline again
    baseline_second, _, _ = time_baseline()
    baseline_ms_per_fs = min(baseline_first, baseline_second)

    mr = [row for row in log if row["phase"] == "multirate"]
    stale = np.array([row["staleness_meV"] for row in mr]) / np.sqrt(3.0)
    ks = np.array([row["k"] for row in mr])
    frozen = [row for row in mr if row["t_fs"] > status["freeze_at"]]
    mr_ms_per_fs = multirate_s / max(mr_fs + fallback_fs, 1) * 1e3

    def frozen_ms_per_fs() -> float | None:
        """Cost per fs of the frozen steady state, from the log's own wall clock.

        ``multirate_ms_per_fs`` averages the training phase in, and training costs
        about 3.5x what frozen running does, so the phase figure answers "what did
        this run cost", not "what does multirate cost once the student is trained".
        Comparing it against a fidelity run, whose student is frozen throughout,
        understated the same loop by 26% until the phases were separated
        (plan section 1.14).
        """
        after = [row for row in mr if row["t_fs"] > status["freeze_at"]]
        if len(after) < 2:
            return None
        wall = after[-1]["wall_s"] - after[0]["wall_s"]
        fs_done = sum(row["k"] for row in after[1:])
        return wall / fs_done * 1e3 if fs_done and wall > 0 else None

    frozen_ms = frozen_ms_per_fs()
    summary = {
        "run": args.run,
        "n_atoms": n,
        "config": {key: v for key, v in vars(args).items() if key not in ("device", "threads")},
        "student": student.config,
        "kernels": {"teacher_cueq": args.teacher_cueq, "teacher_graph": args.teacher_graph,
                    "student_backend": student.config.get("backend", "e3nn"),
                    "neighbours": args.neighbours},
        "staleness_units": unit,
        "skin_list_max_abs_force_error_eV_per_A": skin_error,
        "baseline_ms_per_fs": baseline_ms_per_fs,
        "baseline_ms_per_fs_windows": [baseline_first, baseline_second],
        "bootstrap_fs_used": bootstrap_fs_used,
        "bootstrap_ms_per_fs": bootstrap_s / max(bootstrap_fs_used, 1) * 1e3,
        "multirate_ms_per_fs": mr_ms_per_fs,
        "speedup_multirate_phase": baseline_ms_per_fs / mr_ms_per_fs,
        "frozen_ms_per_fs": frozen_ms,
        "speedup_after_freeze": baseline_ms_per_fs / frozen_ms if frozen_ms else None,
        "speedup_overall": baseline_ms_per_fs * args.fs / ((bootstrap_s + multirate_s) * 1e3),
        "fallback_fs": fallback_fs,
        "ensemble": args.ensemble,
        "drift_gain_budget_meV_per_atom": monitor.gain_budget,
        "drift_horizon_ps": monitor.horizon_ps,
        "drift_refusals": drift_refusals["count"],
        "drift_peak_projection": drift_refusals["peak"],
        "drift_refused_rungs": drift_refusals["rungs"],
        "warm_started_from": None if warm is None else asdict(warm),
        "k_ceiling_after_refusals": controller.ceiling,
        "drift_last_refusal_fs": drift_refusals["last"],
        "mean_k": float((ks * ks).sum() / max(ks.sum(), 1)) if ks.size else None,
        "mean_staleness_meV": float(stale.mean()) if stale.size else None,
        "p90_staleness_meV": float(np.percentile(stale, 90)) if stale.size else None,
        "fraction_outer_steps_over_budget": float((stale > args.epsilon).mean()) if stale.size else None,
        "frozen_at_fs": status["freeze_at"],
        "frozen_by": status["frozen_by"],
        "mean_k_after_freeze": float(np.mean([row["k"] for row in frozen])) if frozen else None,
        "frozen_fs": int(sum(row["k"] for row in frozen)),
        "clock_s": clock,
        "calls": {key: calls_after[key] - calls_before[key] for key in calls},
        "baseline_T": baseline_T,
        "final_T": state.temperature(),
        "baseline_epot_eV_per_atom": baseline_epot,
        "multirate_epot_eV_per_atom": float(torch.stack(mr_epot).mean()) / n if mr_epot else None,
        "log": log,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out = OUT_DIR / f"online_{args.run}{suffix}.json"
    out.write_text(json.dumps(summary, indent=2))

    if not args.no_warm_controller and mr_fs:
        # evidence is what the run held AFTER its last refusal: a ceiling reached
        # while still descending is not one this run certified. certified.lookup
        # applies its own min_hold_fs to that number.
        since = drift_refusals["last"] if drift_refusals["last"] is not None else bootstrap_fs_used
        certified.record(args.certified_store, certified.Certified(
            run=args.run, student=args.warm_start or "", k=int(controller.k),
            ceiling=int(controller.ceiling if controller.ceiling is not None else controller.k_max),
            period_fs=args.fastest_period_fs, epsilon=args.epsilon,
            gain_budget=monitor.gain_budget, evidence_fs=float(t_fs - since),
            refusals=drift_refusals["count"], projection=float(drift_refusals["peak"]),
            refused=drift_refusals["rungs"]))
    print(
        f"{args.run}{suffix}: baseline {baseline_ms_per_fs:.1f} ms/fs (windows "
        f"{baseline_first:.1f}, {baseline_second:.1f}) | multirate {mr_ms_per_fs:.1f} ms/fs "
        f"({summary['speedup_multirate_phase']:.2f}x) | mean k {summary['mean_k']} | staleness mean "
        f"{summary['mean_staleness_meV']} p90 {summary['p90_staleness_meV']} {unit} | over budget "
        f"{summary['fraction_outer_steps_over_budget']} | skin check {skin_error:.1e} | -> {out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
