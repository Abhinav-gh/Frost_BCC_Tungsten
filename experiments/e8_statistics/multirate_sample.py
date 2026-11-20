"""E8: Frost's multirate samplers, judged by the gates that admitted the teacher's step.

The statistics contract is lost at about 0.38x (plan section 1.15) against a
teacher that samples at 4 fs. The exact sampler must keep its drafts accurate
enough to be accepted, while the baseline only has to sample the right
distribution (frost-87). A multirate Frost sampler belongs to the second class: it
is not exact, so it is admitted the way the teacher's 4 fs step was -- by bias
gates against the 1 fs reference (mean teacher energy, its fluctuations, the RDF,
the Si-O-Si angles) -- and then competes on effective samples per second.

A gate is only evidence if it can fail. At a long enough interval a multirate
sampler tends to the *student's* distribution for every mode the residual cannot
correct: on a model problem with a student 5% too stiff, Langevin impulse RESPA
carries 80-100% of the student's bias from H = T/2 on. So the student alone is the
arm every gate must reject, and it is run first. A gate that passes it cannot
certify anything else, because it cannot tell Frost from not running the teacher.

Samplers (--sampler):
  student  BAOAB on the student alone. The teacher only scores frames, and that
           time is excluded from the wall clock.
  respa    Langevin impulse r-RESPA: --k student steps of --inner-dt per teacher
           call, the residual F_T - F_S applied as half impulses at both ends, BAOAB
           inside. This is the dynamics contract's integrator, thermostatted.
  sinr     SIN(R) (frost.isokinetic, XI-RESPA) on the same split. It is
           resonance-free, but on the model problem it is biased near omega H = m pi.
  teacher  BAOAB on the teacher alone, the baseline, through the same code path
           so that matched-step comparisons share everything but the forces.

The teacher energy per atom is recorded at every teacher call (respa, sinr) or
scored frame (student). Frames are kept on a simulated-time cadence. Each run also
probes the teacher's own 4 fs BAOAB step on the same server load, so its ESS/s
can be set against the baseline without trusting a rate measured hours earlier.

With --angle-scalar, every arm also records one configurational scalar -- the mean
bond angle, in multirate_gate.py's --angle convention -- at every inner step. The
energy is only known at teacher calls, so a sampler that calls the teacher every H
fs has an energy series sampled every H fs, and the autocorrelation estimate
cannot resolve anything shorter (g >= 1 floors tau at H). A scalar taken at the
same 4 fs cadence in every arm, the teacher included, measures every sampler's
decorrelation with one instrument (frost-87, frost-ec). It is computed on the
device with no host sync, so it costs the loop almost nothing.

Output: data/results/e8/multirate_<run>_<tag>.json, and the frames as
data/results/e8/<tag>.npz, which angle_gate.py reads.
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


def angle_scalar(spec: str, numbers: np.ndarray, cell: np.ndarray, x0: torch.Tensor, stat: str = "mean"):
    """A function of positions giving the mean bond angle, as a device scalar.

    The same angle as multirate_gate.frame_angles (centre, neighbour species and
    cutoff from ``spec``; self-pairs excluded), for an orthorhombic cell. Each
    centre keeps its nearest neighbours up to one more than the largest
    coordination at x0, and every pair of them within the cutoff contributes.
    """
    from multirate_gate import parse_angle

    centres, neighbours, cutoff = parse_angle(spec, numbers)
    dev, dtype = x0.device, x0.dtype
    c = torch.as_tensor(centres, device=dev)
    nb = torch.as_tensor(neighbours, device=dev)
    lengths = torch.as_tensor(np.diag(cell).copy(), device=dev, dtype=dtype)

    def near_vectors(x):
        d = x[nb][None, :, :] - x[c][:, None, :]
        d = d - lengths * torch.round(d / lengths)
        r = d.norm(dim=2)
        return d, (r < cutoff) & (r > 1e-6)

    k = int(near_vectors(x0)[1].sum(1).max()) + 1
    iu = torch.triu_indices(k, k, 1, device=dev)

    def scalar(x: torch.Tensor) -> torch.Tensor:
        d, near = near_vectors(x)
        score, idx = torch.topk(near.to(dtype), k, dim=1)
        u = torch.gather(d, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        u = u / u.norm(dim=2, keepdim=True).clamp_min(1e-12)
        cos = (u[:, iu[0]] * u[:, iu[1]]).sum(-1)
        valid = ((score[:, iu[0]] > 0) & (score[:, iu[1]] > 0)).to(dtype)
        angles = torch.rad2deg(torch.arccos(cos.clamp(-1.0, 1.0)))
        mean = (angles * valid).sum() / valid.sum()
        if stat == "mean":
            return mean
        # the width, for centres whose mean angle geometry pins (tetrahedra)
        return torch.sqrt(((angles - mean) ** 2 * valid).sum() / valid.sum())

    return scalar


def block_mean(series: np.ndarray, blocks: int = 10) -> tuple[float, float]:
    """Mean and its standard error from non-overlapping blocks."""
    parts = np.array_split(np.asarray(series, dtype=float), blocks)
    means = np.array([p.mean() for p in parts if len(p)])
    return float(means.mean()), float(means.std(ddof=1) / np.sqrt(len(means)))


def main() -> int:
    import h5py
    from mace.calculators import mace_mp

    from frost.integrators import Integrator, Langevin, MDState
    from frost.isokinetic import SINR
    from frost.neighbours import VerletList
    from frost.observables import rdf, tau_int
    from frost.student import GraphedStudent
    from frost.students import load_student
    from frost.teacher import frame_atoms
    from frost.teacher_fast import GraphedTeacher

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="silica1152_T300")
    ap.add_argument("--student", required=True)
    ap.add_argument("--sampler", choices=("student", "respa", "sinr", "teacher"), required=True)
    ap.add_argument("--inner-dt", type=float, default=1.0, help="student step, fs")
    ap.add_argument("--k", type=int, default=1, help="student steps per teacher call (respa, sinr)")
    ap.add_argument("--friction", type=float, default=0.05, help="per fs, BAOAB (student, respa)")
    ap.add_argument("--sinr-L", type=int, default=4)
    ap.add_argument("--sinr-tau-fs", type=float, default=10.0, help="Q1 = Q2 = kT tau^2")
    ap.add_argument("--sinr-gamma", type=float, default=0.1, help="per fs, on the thermostat variables")
    ap.add_argument("--sim-ps", type=float, default=50.0, help="simulated time after burn-in")
    ap.add_argument("--burn-ps", type=float, default=2.0)
    ap.add_argument("--frame-every-fs", type=float, default=50.0)
    ap.add_argument("--score-every-fs", type=float, default=16.0, help="student sampler only")
    ap.add_argument("--probe-steps", type=int, default=300,
                    help="teacher BAOAB steps at 4 fs timed on this run's load, before and after")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--start", type=int, default=9000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True, help="names both outputs; must be new")
    ap.add_argument("--angle-scalar", default=None,
                    help="centre_Z:neighbour_Z:cutoff_A; record the mean bond angle every inner step")
    ap.add_argument("--angle-scalar-stat", choices=("mean", "sd"), default="mean",
                    help="the scalar's statistic; sd for tetrahedral centres, whose mean is pinned")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dev = args.device
    temperature = float(args.run.split("_T")[1])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"multirate_{args.run}_{args.tag}.json"
    frames_out = OUT_DIR / f"{args.tag}.npz"
    # two runs sharing a tag write the same files and one silently wins (trap 20)
    for path in (out, frames_out):
        if path.exists():
            raise SystemExit(f"{path.name} exists; pick a new --tag")
    if args.sampler in ("student", "teacher") and args.k != 1:
        raise SystemExit(f"--k has no meaning for the {args.sampler} sampler")

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

    def probe_teacher() -> float:
        """Teacher BAOAB steps per second at 4 fs, friction 0.05: the baseline's own step."""
        state = MDState(x0.clone(), v0.clone(), masses)
        integ = Integrator(state, 4.0, Langevin(4.0, 0.05, temperature, masses, args.seed + 7))
        _, f, state.x = teacher_force(state.x)
        for _ in range(20):
            _, f = integ.inner_step(teacher_force, f)
        t0 = now()
        for _ in range(args.probe_steps):
            _, f = integ.inner_step(teacher_force, f)
        return args.probe_steps / (now() - t0)

    probe_before = probe_teacher()

    dt = args.inner_dt
    k = args.k
    burn_fs, total_fs = 1000.0 * args.burn_ps, 1000.0 * (args.burn_ps + args.sim_ps)
    energies, frames = [], []
    next_frame = burn_fs
    excluded = 0.0
    t_fs = 0.0

    scalar = (angle_scalar(args.angle_scalar, numbers, cell, x0, args.angle_scalar_stat)
              if args.angle_scalar else None)
    if scalar is not None and args.sampler == "sinr":
        raise SystemExit("--angle-scalar is not wired into the SIN(R) loop")
    scalars: list[torch.Tensor] = []

    def record(x: torch.Tensor) -> None:
        # after burn-in only; appended as a device tensor, synchronised once at the end
        if scalar is not None and t_fs >= burn_fs:
            scalars.append(scalar(x))

    def keep(x: torch.Tensor, e_total: float | None) -> None:
        nonlocal next_frame
        if t_fs < burn_fs:
            return
        if e_total is not None:
            energies.append(e_total / n)
        if t_fs >= next_frame:
            frames.append(x.to(torch.float32).cpu())
            next_frame += args.frame_every_fs

    t0 = now()
    if args.sampler == "student":
        state = MDState(x0.clone(), v0.clone(), masses)
        integ = Integrator(state, dt, Langevin(dt, args.friction, temperature, masses, args.seed))
        _, f_s, state.x = student_force(state.x)
        every = max(1, round(args.score_every_fs / dt))
        step = 0
        while t_fs < total_fs:
            _, f_s = integ.inner_step(student_force, f_s)
            step += 1
            t_fs = step * dt
            record(state.x)
            if step % every == 0 and t_fs >= burn_fs:
                # scoring is measurement, not sampling: its time is not the arm's cost
                s0 = now()
                e_t = float(teacher_energy(state.x))
                excluded += now() - s0
                keep(state.x, e_t)
            elif t_fs >= next_frame:
                keep(state.x, None)
        ensemble = {"friction_per_fs": args.friction}
    elif args.sampler == "teacher":
        state = MDState(x0.clone(), v0.clone(), masses)
        integ = Integrator(state, dt, Langevin(dt, args.friction, temperature, masses, args.seed))
        e_t, f_t, state.x = teacher_force(state.x)
        step = 0
        while t_fs < total_fs:
            e_t, f_t = integ.inner_step(teacher_force, f_t)
            step += 1
            t_fs = step * dt
            record(state.x)
            keep(state.x, float(e_t))
        ensemble = {"friction_per_fs": args.friction}
    elif args.sampler == "respa":
        state = MDState(x0.clone(), v0.clone(), masses)
        integ = Integrator(state, dt, Langevin(dt, args.friction, temperature, masses, args.seed))
        e_t, f_t, state.x = teacher_force(state.x)
        _, f_s, state.x = student_force(state.x)
        slow = f_t - f_s
        outer = 0
        while t_fs < total_fs:
            _, f_s = integ.respa_outer(student_force, f_s, slow, k, on_step=lambda: record(state.x))
            e_t, f_t, state.x = teacher_force(state.x)
            slow = f_t - f_s  # both forces at the same positions (trap 8)
            integ.kick(slow, 0.5 * k * integ.dt)
            outer += 1
            t_fs = outer * k * dt
            keep(state.x, float(e_t))
        ensemble = {"friction_per_fs": args.friction}
    else:
        sampler = SINR.from_ase(masses, temperature, dt, k, L=args.sinr_L, tau_fs=args.sinr_tau_fs,
                                gamma_per_fs=args.sinr_gamma, seed=args.seed)
        sampler.start(x0.clone())
        _, f_s, _ = student_force(sampler.x)
        e_t, f_t, _ = teacher_force(sampler.x)
        slow = f_t - f_s
        last = {"e": float(e_t)}

        def fast(x):
            return student_force(x)[1]

        def slow_at(x, f_fast):
            e, f, _ = teacher_force(x)
            last["e"] = float(e)
            return f - f_fast

        outer = 0
        worst = 0.0
        while t_fs < total_fs:
            f_s, slow = sampler.outer_step(fast, f_s, slow, slow_at)
            outer += 1
            t_fs = outer * k * dt
            keep(sampler.x, last["e"])
            if outer % 50 == 0:
                worst = max(worst, sampler.constraint_residual())
        ensemble = {"L": args.sinr_L, "tau_fs": args.sinr_tau_fs, "gamma_per_fs": args.sinr_gamma,
                    "scheme": sampler.scheme, "worst_constraint_residual": worst}
    seconds = now() - t0 - excluded

    probe_after = probe_teacher()

    kept_fraction = args.sim_ps / (args.sim_ps + args.burn_ps)
    scalar_report = None
    if scalars:
        s_series = torch.stack(scalars).double().cpu().numpy()
        s_tau, s_tau_se = tau_int(s_series)
        s_ess = len(s_series) / s_tau
        s_rate = s_ess / (seconds * kept_fraction)
        scalar_report = {"spec": args.angle_scalar, "interval_fs": dt, "samples": len(s_series),
                         "mean_deg": float(s_series.mean()), "tau_int_samples": s_tau,
                         "tau_int_se": s_tau_se, "tau_int_fs": s_tau * dt, "ess": s_ess,
                         "ess_per_kept_second": s_rate, "ess_per_kept_second_se": s_rate * s_tau_se / s_tau}
    series = np.asarray(energies)
    mean, se = block_mean(series)
    tau, tau_se = tau_int(series)
    ess = len(series) / tau
    sample_fs = dt if args.sampler != "student" and k == 1 else (
        k * dt if args.sampler != "student" else max(1, round(args.score_every_fs / dt)) * dt)
    n_species = int(types.max()) + 1
    types_np = types.cpu().numpy()
    half = len(frames) // 2
    g = np.asarray(rdf(torch.stack(frames), cell, types_np, n_species)["g_total"])
    g_a = np.asarray(rdf(torch.stack(frames[:half]), cell, types_np, n_species)["g_total"])
    g_b = np.asarray(rdf(torch.stack(frames[half:]), cell, types_np, n_species)["g_total"])
    report = {
        "run": args.run, "n_atoms": n, "arms": "multirate", "sampler": args.sampler,
        "student": student.config,
        "config": {key: v for key, v in vars(args).items() if key != "device"},
        "ensemble": ensemble,
        "multirate": {
            "simulated_ps": args.sim_ps, "wall_seconds": seconds,
            "excluded_scoring_seconds": excluded,
            "ms_per_simulated_fs": 1000.0 * seconds / (total_fs - 0.0),
            "samples": len(series), "sample_interval_fs": sample_fs,
            "epot_mean_eV_per_atom": mean, "epot_mean_se": se,
            "epot_var_per_atom": float(np.var(series)),
            "tau_int_samples": tau, "tau_int_se": tau_se, "tau_int_fs": tau * sample_fs,
            "ess": ess, "ess_per_second": ess / seconds,
            "ess_per_second_se": ess / seconds * tau_se / tau,
            "rdf": {"g_total": g.tolist()},
            "rdf_halves_l1": float(np.abs(g_b - g_a).sum() / np.abs(g_a).sum()),
            "frames": len(frames),
            "angle_scalar": scalar_report,
        },
        "teacher_probe_steps_per_second": {"before": probe_before, "after": probe_after},
    }
    np.savez(frames_out, frames=torch.stack(frames).numpy(), energies=series,
             seconds=seconds, steps=len(series),
             angle_scalar=np.asarray([]) if scalar_report is None else s_series)
    out.write_text(json.dumps(report, indent=2))
    m = report["multirate"]
    print(f"{args.run} {args.sampler} dt {dt} k {k}: Epot {mean:.6f} +/- {se:.6f} eV/atom | var "
          f"{m['epot_var_per_atom']:.3e} | tau {tau:.1f} samples ({m['tau_int_fs']:.0f} fs) | ESS/s "
          f"{m['ess_per_second']:.3f} +/- {m['ess_per_second_se']:.3f} | {m['ms_per_simulated_fs']:.2f} "
          f"ms/fs | RDF halves {m['rdf_halves_l1']:.4f} | frames {len(frames)} | teacher probe "
          f"{probe_before:.1f}/{probe_after:.1f} steps/s | -> {out.name}, {frames_out.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
