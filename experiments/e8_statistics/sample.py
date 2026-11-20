"""E8 (gate G4): exact teacher statistics from student drafts, scored per wall second.

Water refuses under the dynamics contract (plan section 1.13), but many water
workloads need equilibrium averages, not trajectories. This runs two samplers of
the teacher's canonical distribution for the same wall-clock budget:

  teacher  BAOAB Langevin MD with the CUDA-graph teacher (the E1 settings);
  nested   ``frost.nested.NestedHMC``: the CUDA-graph student drafts
           ``--n-draft`` velocity-Verlet steps, one energy-only teacher call
           accepts or rejects, and the momenta are refreshed.

Two questions, answered separately:

- **Exactness.** Mean and variance of the teacher potential energy per atom, and
  the radial distribution functions, must agree within block-averaged error bars.
  Nested HMC is exact by construction; this checks the implementation on the real
  system.
- **Efficiency.** Effective samples per wall second of the teacher potential energy,
  from its integrated autocorrelation time (Sokal's windowing). The ratio of the two
  is the statistics speedup S_stat.

Output: data/results/e8/sample_<run>[_<tag>].json
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


def block_mean(series: np.ndarray, blocks: int = 10) -> tuple[float, float]:
    """Mean and its standard error from non-overlapping blocks."""
    parts = np.array_split(np.asarray(series, dtype=float), blocks)
    means = np.array([p.mean() for p in parts if len(p)])
    return float(means.mean()), float(means.std(ddof=1) / np.sqrt(len(means)))


def main() -> int:
    import h5py
    from mace.calculators import mace_mp

    from frost.integrators import Integrator, Langevin, MDState
    from frost.neighbours import VerletList
    from frost.nested import NestedHMC
    from frost.observables import ratio_se, rdf, tau_int
    from frost.student import GraphedStudent
    from frost.students import load_student
    from frost.teacher import frame_atoms
    from frost.teacher_fast import GraphedTeacher

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="water216_T300")
    ap.add_argument("--student", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--start", type=int, default=9000)
    ap.add_argument("--budget-s", type=float, default=180.0, help="wall seconds, nested arm")
    ap.add_argument("--teacher-budget-s", type=float, default=None,
                    help="wall seconds, teacher arm (default: --budget-s). The teacher's tau is "
                         "the longer one, so its arm dominates the error on S_stat.")
    ap.add_argument("--dt", type=float, default=1.0, help="teacher arm's step, fs")
    ap.add_argument("--draft-dt", type=float, default=None,
                    help="student draft step, fs (default: --dt). Separate, because the baseline "
                         "must keep its own step: the draft's integration error only lowers "
                         "acceptance, while the teacher arm's would change what it samples")
    ap.add_argument("--friction", type=float, default=0.01, help="per fs, teacher arm")
    ap.add_argument("--n-draft", type=int, default=8)
    ap.add_argument("--refresh", type=float, default=1.0)
    ap.add_argument("--extra-chances", type=int, default=0,
                    help="further drafts tested after a rejection (frost.nested)")
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="draft length drawn from n_draft * [1 - jitter, 1 + jitter]")
    ap.add_argument("--arms", choices=("both", "teacher"), default="both",
                    help="'teacher' runs and reports the baseline arm alone, which is how its "
                         "friction is tuned before it is used as B0")
    ap.add_argument("--teacher-ref", default=None,
                    help="stored teacher arm (.npz) to compare against instead of rerunning it; "
                         "its throughput is re-probed and the faster rate is used (trap 13)")
    ap.add_argument("--ref-out", default=None,
                    help="filename for --save-teacher-ref (default teacher_ref_<run>.npz). The "
                         "default never depends on --tag: queued jobs refer to it by name")
    ap.add_argument("--save-teacher-ref", action="store_true",
                    help="store this run's teacher arm for later --teacher-ref comparisons")
    ap.add_argument("--probe-steps", type=int, default=300,
                    help="teacher steps timed when --teacher-ref is given")
    ap.add_argument("--rdf-every-s", type=float, default=2.0,
                    help="wall seconds between RDF frames, teacher arm (every step costs the same)")
    ap.add_argument("--rdf-every-proposals", type=int, default=10,
                    help="proposals between RDF frames, nested arm. Never wall time: a proposal's "
                         "cost depends on its state (chances used, draft length), so frames taken "
                         "on a clock weight each state by its sojourn time and bias the RDF")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
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
    x0 = torch.as_tensor(atoms.get_positions(), dtype=torch.float64, device=dev)
    v0 = torch.as_tensor(velocities, dtype=torch.float64, device=dev)

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

    def teacher_force(x):
        if t_list.update(x)[1]:
            teacher.set_edges(t_list.i, t_list.j, t_list.shifts, t_list.unit_shifts)
        e, f = teacher(x)
        return e, f.double(), x

    def teacher_energy(x):
        if t_list.update(x)[1]:
            teacher.set_edges(t_list.i, t_list.j, t_list.shifts, t_list.unit_shifts)
        return teacher.energy(x)

    def student_force(x):
        if s_list.update(x)[1]:
            graphed.set_edges(s_list.i, s_list.j, s_list.shifts)
        e, f = graphed(x)
        return e.detach().sum(), f.detach().double(), x

    def now():
        torch.cuda.synchronize()
        return time.perf_counter()

    n_species = int(types.max()) + 1
    types_np = types.cpu().numpy()

    # ------------------------------------------------------------ teacher arm
    state = MDState(x0.clone(), v0.clone(), masses)
    integ = Integrator(state, args.dt, Langevin(args.dt, args.friction, temperature, masses, args.seed))
    e, f, state.x = teacher_force(state.x)
    if args.teacher_ref:
        # the stored arm supplies the energy series and frames; only its wall time is
        # re-measured, on this run's server load, and the faster rate is kept
        if not Path(args.teacher_ref).exists():
            have = sorted(q.name for q in OUT_DIR.glob("teacher_ref_*.npz"))
            raise SystemExit(f"no teacher reference at {args.teacher_ref}. Available: "
                             f"{have or 'none'} (make one with --save-teacher-ref)")
        ref = np.load(args.teacher_ref)
        t0 = now()
        for _ in range(args.probe_steps):
            e, f = integ.inner_step(teacher_force, f)
        probe_rate = args.probe_steps / (now() - t0)
        stored_rate = float(ref["steps"]) / float(ref["seconds"])
        rate = max(probe_rate, stored_rate)
        teacher_arm = {"energies": ref["energies"], "frames": list(torch.from_numpy(ref["frames"])),
                       "seconds": float(ref["steps"]) / rate, "steps": int(ref["steps"]),
                       "stored_rate": stored_rate, "probe_rate": probe_rate, "ref": args.teacher_ref}
    else:
        energies, frames = [], []
        t0 = now()
        next_rdf = 0.0
        steps = 0
        teacher_budget = args.teacher_budget_s or args.budget_s
        while (elapsed := now() - t0) < teacher_budget:
            e, f = integ.inner_step(teacher_force, f)
            energies.append(float(e) / n)
            steps += 1
            if elapsed >= next_rdf:
                frames.append(state.x.to(torch.float32).cpu())
                next_rdf += args.rdf_every_s
        teacher_seconds = now() - t0
        teacher_arm = {"energies": np.array(energies), "frames": frames, "seconds": teacher_seconds,
                       "steps": steps}
        if args.save_teacher_ref:
            # --tag means this is one arm among several, not the canonical
            # reference. Four arms with distinct tags once wrote the same path in
            # turn and the survivor -- a 20-minute dt=4 arm -- stood in for the
            # 60-minute dt=1 reference for 100 minutes. Refuse the combination
            # rather than warn: a warning in a queued job's log is a warning
            # nobody reads.
            if args.tag and not args.ref_out:
                raise SystemExit("--save-teacher-ref with --tag needs an explicit --ref-out: "
                                 "tagged arms are candidates, not the canonical reference")
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            ref_path = OUT_DIR / (args.ref_out or f"teacher_ref_{args.run}.npz")
            if ref_path.exists() and not args.ref_out:
                raise SystemExit(f"{ref_path.name} exists; pass --ref-out to write elsewhere "
                                 "(the canonical reference is shared across sessions)")
            np.savez(ref_path, energies=teacher_arm["energies"], frames=torch.stack(frames).numpy(),
                     seconds=teacher_seconds, steps=steps)
            print(f"teacher reference -> {ref_path}", flush=True)

    # ------------------------------------------------------------ comparison
    def halves_l1(frames) -> float:
        if len(frames) < 4:
            return float("nan")
        half = len(frames) // 2
        a = np.asarray(rdf(torch.stack(frames[:half]), cell, types_np, n_species)["g_total"])
        b = np.asarray(rdf(torch.stack(frames[half:]), cell, types_np, n_species)["g_total"])
        return float(np.abs(b - a).sum() / np.abs(a).sum())

    def summarise(arm, burn: float = 0.1):
        series = arm["energies"][int(burn * len(arm["energies"])):]
        # the frames get the same burn-in as the energies. Every run starts from the
        # same configuration, so keeping the first frames imprints that one structure
        # on every run alike, which looks exactly like a shared bias
        kept = arm["frames"][int(burn * len(arm["frames"])):]
        mean, se = block_mean(series)
        tau, tau_se = tau_int(series)
        ess = len(series) / tau
        ess_per_second = ess / arm["seconds"]
        return {
            "samples": len(series), "wall_seconds": arm["seconds"], "steps_or_proposals": arm["steps"],
            "epot_mean_eV_per_atom": mean, "epot_mean_se": se,
            "epot_var_per_atom": float(np.var(series)),
            "tau_int_samples": tau, "tau_int_se": tau_se, "ess": ess,
            "ess_per_second": ess_per_second, "ess_per_second_se": ess_per_second * tau_se / tau,
            "rdf": rdf(torch.stack(kept), cell, types_np, n_species),
            # this arm's own half-to-half difference: the noise floor for its own frames
            "rdf_halves_l1": halves_l1(kept),
            "frames": len(kept),
        }

    if args.arms == "teacher":
        t = summarise(teacher_arm)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"sample_{args.run}{'_' + args.tag if args.tag else ''}.json"
        out.write_text(json.dumps({"run": args.run, "n_atoms": n, "arms": "teacher",
                                   "config": {k: v for k, v in vars(args).items() if k != "device"},
                                   "teacher": t}, indent=2))
        print(f"{args.run} teacher only: friction {args.friction}/fs | ESS/s "
              f"{t['ess_per_second']:.3f} +/- {t['ess_per_second_se']:.3f} | tau "
              f"{t['tau_int_samples']:.0f} +/- {t['tau_int_se']:.0f} samples | steps "
              f"{t['steps_or_proposals']} | -> {out}", flush=True)
        return 0

    # ------------------------------------------------------------ nested arm
    x, v = x0.clone(), v0.clone()
    e_t = float(teacher_energy(x))
    f_s = student_force(x)[1]
    sampler = NestedHMC(masses, temperature, dt_fs=args.draft_dt or args.dt, n_draft=args.n_draft,
                        student_force=student_force, teacher_energy=teacher_energy,
                        refresh=args.refresh, extra_chances=args.extra_chances,
                        jitter=args.jitter, seed=args.seed + 1)
    energies, frames = [], []
    t0 = now()
    while now() - t0 < args.budget_s:
        x, v, e_t, f_s, _ = sampler.step(x, v, e_t, f_s)
        energies.append(e_t / n)
        if sampler.proposed % args.rdf_every_proposals == 0:
            frames.append(x.to(torch.float32).cpu())
    nested_seconds = now() - t0
    nested_arm = {"energies": np.array(energies), "frames": frames, "seconds": nested_seconds,
                  "steps": sampler.proposed}

    t, s = summarise(teacher_arm), summarise(nested_arm)
    g_t, g_s = np.asarray(t["rdf"]["g_total"]), np.asarray(s["rdf"]["g_total"])
    kept_t = teacher_arm["frames"][int(0.1 * len(teacher_arm["frames"])):]
    half = len(kept_t) // 2
    g_a = np.asarray(rdf(torch.stack(kept_t[:half]), cell, types_np, n_species)["g_total"])
    g_b = np.asarray(rdf(torch.stack(kept_t[half:]), cell, types_np, n_species)["g_total"])
    report = {
        "run": args.run, "n_atoms": n, "student": student.config,
        "config": {k: v for k, v in vars(args).items() if k != "device"},
        "teacher": t, "nested": s,
        "acceptance": sampler.acceptance,
        "first_chance_acceptance": float(np.mean([r["accepted"] and r["chances"] == 1
                                                  for r in sampler.history])),
        "mean_chances": float(np.mean([r["chances"] for r in sampler.history])),
        "mean_draft_fs": float(np.mean([r["draft_fs"] for r in sampler.history])),
        "teacher_rate": {key: teacher_arm[key] for key in ("stored_rate", "probe_rate", "ref")
                         if key in teacher_arm},
        "exactness": {
            "epot_mean_difference_in_se": (s["epot_mean_eV_per_atom"] - t["epot_mean_eV_per_atom"])
            / float(np.hypot(s["epot_mean_se"], t["epot_mean_se"])),
            "rdf_l1_nested_vs_teacher": float(np.abs(g_s - g_t).sum() / np.abs(g_t).sum()),
            "rdf_l1_teacher_halves": float(np.abs(g_b - g_a).sum() / np.abs(g_a).sum()),
            "rdf_l1_nested_halves": s["rdf_halves_l1"],
        },
        "S_stat": s["ess_per_second"] / t["ess_per_second"],
        "S_stat_se": ratio_se(s["ess_per_second"], s["ess_per_second_se"],
                              t["ess_per_second"], t["ess_per_second_se"]),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out = OUT_DIR / f"sample_{args.run}{suffix}.json"
    out.write_text(json.dumps(report, indent=2))
    ex = report["exactness"]
    print(f"{args.run}{suffix}: S_stat {report['S_stat']:.2f} +/- {report['S_stat_se']:.2f}x | acceptance {sampler.acceptance:.2f} | "
          f"ESS/s teacher {t['ess_per_second']:.2f} nested {s['ess_per_second']:.2f} | tau "
          f"{t['tau_int_samples']:.0f} vs {s['tau_int_samples']:.1f} samples | Epot "
          f"{t['epot_mean_eV_per_atom']:.5f} vs {s['epot_mean_eV_per_atom']:.5f} "
          f"({ex['epot_mean_difference_in_se']:+.1f} se) | RDF L1 {ex['rdf_l1_nested_vs_teacher']:.3f} "
          f"(teacher halves {ex['rdf_l1_teacher_halves']:.3f}) | -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
