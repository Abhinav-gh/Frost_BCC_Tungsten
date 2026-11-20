"""Spatial multirate on an interface (further_expansions_beyond.md 7.3; frost.spatial).

    python experiments/e19_spatial/spatial_runtime.py --run cuwater_T300 --student <.pt> --fs 10000 --tag sp_s0

Arms, all from the same start, NVT (Langevin 1/ps), hydrogens at 4 amu, 1 fs:
- teacher: the full teacher every step (the reference, and its cost);
- global: one interval for every atom, k_fast (what water's fastest period allows everywhere);
- spatial_b<B>: the fast region's residual refreshed every k_fast from a ``RegionTeacher`` (region + B A buffer), the
  whole system's every k_slow = m k_fast from the full teacher; the two residuals are RESPA impulses on their own clocks,
  applied to disjoint atom sets (symplectic, reversible). At every full refresh the region teacher is also evaluated at the
  same positions and its error against the full teacher is recorded (the measured truncation error; not counted in the
  cost: a production run would check it only now and then).
The fast region: every non-Cu atom plus the top ``--interface-layers`` Cu layers.

Records per arm: wall time per fs, teacher calls, atom-evaluations (atoms passed to a teacher), the staleness of each
region's residual, the measured region error, temperatures per region; frames for ``score_frames.py``.
Output: data/results/e19/spatial_<run>_<tag>.json and frames_<run>_<tag>.npz.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from ase import units

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "results" / "e19"


def staleness(dr: torch.Tensor, m: torch.Tensor) -> float:
    """RMS per component of a residual change in acceleration units, meV/(A amu) (MDBackend's staleness)."""
    if dr.numel() == 0:
        return 0.0
    return float(((dr / m) ** 2).sum(1).mean().sqrt()) * 1e3 / np.sqrt(3.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--start", type=int, default=9999)
    ap.add_argument("--fs", type=float, default=10000.0)
    ap.add_argument("--hmr", type=float, default=3.0)
    ap.add_argument("--k-fast", type=int, default=3)
    ap.add_argument("--k-slow", type=int, default=24)
    ap.add_argument("--buffers", default="24:12,8:4", help="region-teacher variants, buffer:inner in A, comma-separated")
    ap.add_argument("--interface-layers", type=int, default=2)
    ap.add_argument("--friction", type=float, default=0.001)
    ap.add_argument("--compile", default="default")
    ap.add_argument("--arms", default="teacher,global,spatial")
    ap.add_argument("--graphed-region", action="store_true",
                    help="region teachers captured in CUDA graphs (frost.spatial.GraphedRegionTeacher), checked against "
                         "the eager ones at the start frame")
    ap.add_argument("--edge-region", action="store_true",
                    help="with --graphed-region: the masked energy's forces through edge-vector gradients summed with "
                         "index_add (frost.mace_compose), not the sort-based backward of the position gather; checked "
                         "against the eager positions-gradient teacher like any graphed one")
    ap.add_argument("--tight-teacher", action="store_true",
                    help="filter the full teacher's and the region teachers' skin lists to r_max before every call")
    ap.add_argument("--seed", type=int, default=0, help="the Langevin noise seed (independent realisations, trap 30)")
    ap.add_argument("--dt", type=float, default=1.0, help="fs; frames stay 4 fs (v) and 20 fs (x) apart at any step")
    ap.add_argument("--check-every", type=int, default=1,
                    help="measure the region error at every n-th full refresh (it runs inside the timed loop: 1, the "
                         "first runs' setting, costs one extra region call per k_slow; a production run checks now and "
                         "then)")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    if args.k_slow % args.k_fast:
        raise SystemExit("--k-slow must be a multiple of --k-fast")
    torch.set_num_threads(4)

    from frost.integrators import MDState
    from frost.runtime.md import build
    from frost.spatial import GraphedRegionTeacher, RegionTeacher
    from frost.teachers import load_calc

    b = build(args.run, args.student, start=args.start, dt_fs=args.dt, ensemble="nvt", friction=args.friction,
              hmr=args.hmr, edge_forces=True, compile_student=args.compile, derived_lists=False, seed=args.seed,
              tight_teacher=args.tight_teacher)
    n, masses, atoms = b["n"], b["masses"], b["atoms"]
    dev = b["x0"].device
    numbers = atoms.get_atomic_numbers()
    z = atoms.get_positions()[:, 2]
    cu = numbers == 29
    layers = np.unique(np.round(z[cu], 1))
    z_cut = layers[-args.interface_layers] - 0.3 if args.interface_layers else np.inf
    fast_np = (~cu) | (cu & (z >= z_cut))
    fast = torch.as_tensor(fast_np, device=dev)
    slow = ~fast
    fast_idx = np.flatnonzero(fast_np)
    print(f"{args.run}: {n} atoms, fast region {int(fast_np.sum())} (non-Cu + top {args.interface_layers} Cu layers), "
          f"slow {int((~fast_np).sum())}; k_fast {args.k_fast}, k_slow {args.k_slow}", flush=True)
    calc = load_calc(b["teacher_name"], str(dev), "float32", enable_cueq=True)
    # self-check: with a buffer covering the whole cell and every site energy summed, the region teacher must give the
    # full teacher's forces on the region (the masking and the autograd path, before any run trusts them)
    whole = RegionTeacher(calc, atoms, fast_idx, 1e4, 1e4, str(dev))
    _, f_full, _ = b["teacher_force"](b["x0"].clone())
    diff = float((whole(b["x0"]) - f_full[torch.as_tensor(fast_np, device=dev)]).abs().max())
    print(f"self-check: whole-cell region teacher against the full teacher, max |dF| {diff:.2e} eV/A", flush=True)
    if diff > 2e-3:
        raise SystemExit(f"region teacher disagrees with the full teacher by {diff:.2e} eV/A")
    del whole
    region_teachers = {}
    for spec in [s for s in args.buffers.split(",") if s]:
        buf, inner = (float(v) for v in spec.split(":"))
        rt = RegionTeacher(calc, atoms, fast_idx, buf, inner, str(dev))
        if args.graphed_region:
            g = GraphedRegionTeacher(calc, atoms, fast_idx, buf, inner, str(dev), edge_path=args.edge_region,
                                     tight=args.tight_teacher)

            def rel(x):  # the largest force difference against the largest force, at one geometry
                fe = rt(x)
                return float((g(x) - fe).abs().max()), float(fe.abs().max())

            dg, fg = rel(b["x0"])
            # a displacement whose largest component (~4 sigma over 3,510 components) passes the list's half skin, so
            # the graphed teacher must rewrite its edges in place; a 0.4 A sigma (the first check, 2026-11-15) pushed atoms
            # to ~1 A apart, where float32 summation-order noise between graph and eager reached 7.8e-3 eV/A
            x_kick = b["x0"] + 0.15 * torch.randn_like(b["x0"])
            dk, fk = rel(x_kick)
            print(f"  graphed region teacher b {buf:g} A against the eager one: max |dF| {dg:.2e} of max |F| {fg:.2f} "
                  f"(start), {dk:.2e} of {fk:.2f} (after a rebuild, {g.list.rebuilds} rebuilds) eV/A", flush=True)
            if dg > 2e-3 or dk > 1e-4 * max(fk, 1.0) + 2e-3:
                raise SystemExit(f"graphed region teacher disagrees with the eager one: {dg:.2e} / {dk:.2e} eV/A")
            rt = g
        kind = ("_ge" if args.edge_region else "_g") if args.graphed_region else ""
        region_teachers[f"spatial{kind}_b{buf:g}"] = rt
        print(f"  region teacher b {buf:g} A, inner {inner:g} A: {rt.n_sub} atoms ({rt.n_sub / n:.0%}), "
              f"{rt.n_inner} site energies", flush=True)
    dt = args.dt * units.fs
    steps = round(args.fs / args.dt)
    v_every, x_every, t_every = max(1, round(4.0 / args.dt)), max(1, round(20.0 / args.dt)), max(1, round(100.0 / args.dt))
    report = {"run": args.run, "config": vars(args), "n_atoms": n, "n_fast": int(fast_np.sum()), "arms": {}}
    frames_out = {}

    def recorder():
        rec = {"x": [], "v": [], "T_fast": [], "T_slow": []}

        def on_step(step, st):
            if step % v_every == 0:
                rec["v"].append(st.v.detach().float().clone())
            if step % x_every == 0:
                rec["x"].append(st.x.detach().float().clone())
            if step % t_every == 0:
                ke = 0.5 * (masses * st.v ** 2).sum(1)
                rec["T_fast"].append(float(2 * ke[fast].sum() / (3 * int(fast.sum()) * units.kB)))
                rec["T_slow"].append(float(2 * ke[slow].sum() / (3 * max(1, int(slow.sum())) * units.kB)))
        return rec, on_step

    def finish(name, rec, sec, extra):
        row = {"ms_per_fs": 1e3 * sec / args.fs, "T_fast": float(np.mean(rec["T_fast"])),
               "T_slow": float(np.mean(rec["T_slow"])) if rec["T_slow"] else None, **extra}
        if "teacher" in report["arms"]:
            row["speedup"] = report["arms"]["teacher"]["ms_per_fs"] / row["ms_per_fs"]
        report["arms"][name] = row
        frames_out[f"{name}__x"] = torch.stack(rec["x"]).cpu().numpy()
        frames_out[f"{name}__v"] = torch.stack(rec["v"]).cpu().numpy()
        print(f"{name}: {row['ms_per_fs']:.2f} ms/fs" + (f" ({row['speedup']:.2f}x)" if "speedup" in row else "")
              + f" | T fast {row['T_fast']:.1f} K slow {row['T_slow'] or 0:.1f} K | "
              + " ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in extra.items()
                         if not isinstance(v, list)), flush=True)

    arms = args.arms.split(",")
    if "teacher" in arms:
        st = MDState(b["x0"].clone(), b["v0"].clone(), masses)
        integ = b["integrator"](st)
        rec, on_step = recorder()
        _, f, st.x = b["teacher_force"](st.x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(steps):
            _, f = integ.inner_step(b["teacher_force"], f)
            on_step(i + 1, st)
        torch.cuda.synchronize()
        finish("teacher", rec, time.perf_counter() - t0, {"teacher_calls": steps, "atom_evals": steps * n})

    def run_frost(name, k_fast, k_slow, rt, fast_m, slow_m):
        st = MDState(b["x0"].clone(), b["v0"].clone(), masses)
        integ = b["integrator"](st)
        rec, on_step = recorder()
        _, f_t, st.x = b["teacher_force"](st.x)
        _, f_s, st.x = b["student_force"](st.x)
        res = f_t - f_s
        mf, ms = masses[fast_m], masses[slow_m]
        m = k_slow // k_fast
        full_calls, region_calls, atom_evals = 1, 0, n
        st_fast, st_slow, region_err = [], [], []

        def kick(mask, duration):
            st.v[mask] = st.v[mask] + duration * res[mask] / masses[mask]

        done = 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        while done < steps:
            kick(slow_m, 0.5 * k_slow * dt)
            for j in range(m):
                kick(fast_m, 0.5 * k_fast * dt)
                for _ in range(k_fast):
                    _, f_s = integ.inner_step(b["student_force"], f_s)
                    done += 1
                    on_step(done, st)
                if j < m - 1:  # the fast region alone, from the subsystem
                    new_fast = rt(st.x) - f_s[fast_m]
                    st_fast.append(staleness(new_fast - res[fast_m], mf))
                    res[fast_m] = new_fast
                    region_calls += 1
                    atom_evals += rt.n_sub
                else:  # everything, from the full teacher
                    _, f_t, st.x = b["teacher_force"](st.x)
                    new = f_t - f_s
                    if rt is not None and (full_calls - 1) % args.check_every == 0:
                        region_err.append(staleness(rt(st.x) - f_t[fast_m], mf))  # measured, not counted
                    st_fast.append(staleness(new[fast_m] - res[fast_m], mf))
                    st_slow.append(staleness(new[slow_m] - res[slow_m], ms))
                    res = new
                    full_calls += 1
                    atom_evals += n
                kick(fast_m, 0.5 * k_fast * dt)
            kick(slow_m, 0.5 * k_slow * dt)
        torch.cuda.synchronize()
        sec = time.perf_counter() - t0
        extra = {"full_calls": full_calls, "region_calls": region_calls, "atom_evals": atom_evals,
                 "atom_evals_per_teacher_step": atom_evals / steps / n,
                 "staleness_fast_mean": float(np.mean(st_fast)) if st_fast else 0.0,
                 "staleness_fast_max": float(np.max(st_fast)) if st_fast else 0.0,
                 "staleness_slow_mean": float(np.mean(st_slow)) if st_slow else 0.0,
                 "staleness_slow_max": float(np.max(st_slow)) if st_slow else 0.0}
        if region_err:
            extra.update(region_error_mean=float(np.mean(region_err)), region_error_max=float(np.max(region_err)))
        finish(name, rec, sec, extra)

    all_atoms = torch.ones(n, dtype=torch.bool, device=dev)
    if "global" in arms:
        run_frost("global", args.k_fast, args.k_fast, None, all_atoms, ~all_atoms)
    if "spatial" in arms:
        for name, rt in region_teachers.items():
            run_frost(name, args.k_fast, args.k_slow, rt, fast, slow)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"spatial_{args.run}_{args.tag}.json").write_text(json.dumps(report, indent=1))
    np.savez(OUT / f"frames_{args.run}_{args.tag}.npz", numbers=numbers, cell=np.array(atoms.cell.array),
             masses=masses.squeeze(-1).cpu().numpy(), v_fs=4.0, x_fs=20.0, **frames_out)
    print(f"-> {OUT / f'spatial_{args.run}_{args.tag}.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
