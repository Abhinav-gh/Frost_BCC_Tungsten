"""Frost on radiation-damage cascades (further_expansions_beyond.md 6.9).

A primary knock-on atom (PKA) of energy E is launched from the middle of a BCC crystal; the cascade's ballistic phase
lasts a few hundred fs, the thermal spike a few ps, and the surviving Frenkel pairs are counted once it cools. What it
tests in Frost: the fastest motion is no longer a vibration, so
- the time step adapts (the fastest atom moves at most --xmax per step, between --dt-min and --dt-max), and an interval
  is a duration in fs (k x dt-max), filled by however many adaptive inner steps it takes; the held residual's two
  half kicks use that known duration, and the resonance prior bounds it as a time, as for every other run;
- the ZBL repulsion (frost.zbl) is part of the fast force, exact on every inner step and never held; only the smooth
  many-body part, teacher (MACE-MP-0) minus student, is held;
- a border shell of atoms is thermostatted (Langevin) to absorb the shock wave; the interior is NVE.
Arms from the same equilibrated crystal and PKA: ``teacher`` (MACE + ZBL every step), ``frost`` (student + ZBL every
step, the residual held, interval adapted by the magnitude clause with rollback), ``frostx`` (the same, and an interval
of k 1 that still fails runs on the teacher: the first runs let it stand, and the smoke run found six such intervals in
one 200 eV cascade) and ``frostloc`` (frostx with the clause also applied region by region, frost.locality, 2026-11-14:
a cascade's fast physics lives in a few hundred atoms, which the whole-cell RMS dilutes). Damage: Wigner-Seitz occupancy at the end (drift-corrected, as
experiments/e13_runtime/defects.py) and every --sample-fs with the core temperature (peak damage and its time), and
timings (sampling excluded).

Usage (GPU): python experiments/e16_cascades/cascade_runtime.py --element W --cells 14 --student <w student .pt> \
                 --pka-ev 500 --fs 6000 --seeds 0 1 2 --tag c500
Output: data/results/e16/cascade_<element><cells>_<tag>.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from scipy.spatial import cKDTree

from frost.controller import IntervalController
from frost.locality import staleness_stats
from frost.neighbours import VerletList, tight_edges
from frost.student import GraphedStudent
from frost.students import load_student
from frost.systems import bcc
from frost.teacher_fast import GraphedTeacher
from frost.teachers import load_calc
from frost.zbl import ZBL

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "results" / "e16"


class Forces:
    """Teacher, student and ZBL, each on its own Verlet list (unwrapped positions)."""

    def __init__(self, atoms, student_path: str, dev: str, zbl_in: float, zbl_out: float, tight: bool = False):
        cell = np.array(atoms.cell.array)
        n = len(atoms)
        x0 = torch.as_tensor(atoms.get_positions(), dtype=torch.float64, device=dev)
        self.teacher = GraphedTeacher(load_calc("mp-medium", dev, "float32", enable_cueq=True), atoms, dev, edge_forces=True)
        self.t_list = VerletList(cell, self.teacher.r_max, 1.0, dev, wrap_positions=False, builder="alchemi")
        self.t_list.update(x0, force=True)
        # tight (2026-11-17): the skin list filtered to r_max before every teacher call, as md_runtime --tight-teacher
        # (frost.neighbours.tight_edges); the graph's buffers are sized from the filtered list
        self.tight, self.r_cut2 = tight, float(self.teacher.r_max) ** 2
        self.teacher.set_edges(*(tight_edges(self.t_list, x0, self.r_cut2) if tight
                                 else (self.t_list.i, self.t_list.j, self.t_list.shifts, self.t_list.unit_shifts)))
        st = load_student(student_path, dev)
        types = st.types(torch.as_tensor(atoms.get_atomic_numbers(), device=dev))
        self.s_list = VerletList(cell, float(st.config["cutoff"]), 0.5, dev, wrap_positions=False, builder="alchemi")
        self.s_list.update(x0, force=True)
        self.student = GraphedStudent(st, types, n, capacity=int(1.6 * self.s_list.i.numel()) + 64)
        self.student.set_edges(self.s_list.i, self.s_list.j, self.s_list.shifts)
        self.zbl = ZBL(torch.as_tensor(atoms.get_atomic_numbers(), device=dev), zbl_in, zbl_out)
        self.z_list = VerletList(cell, zbl_out, 0.5, dev, wrap_positions=False, builder="alchemi")
        self.versions = {"t": self.t_list.rebuilds, "s": self.s_list.rebuilds}

    def teacher_f(self, x):
        self.t_list.update(x)
        if self.tight:
            self.teacher.set_edges(*tight_edges(self.t_list, x, self.r_cut2))
        elif self.t_list.rebuilds != self.versions["t"]:
            self.versions["t"] = self.t_list.rebuilds
            self.teacher.set_edges(self.t_list.i, self.t_list.j, self.t_list.shifts, self.t_list.unit_shifts)
        e, f = self.teacher(x)
        return float(e), f.double()

    def student_f(self, x):
        self.s_list.update(x)
        if self.s_list.rebuilds != self.versions["s"]:
            self.versions["s"] = self.s_list.rebuilds
            self.student.set_edges(self.s_list.i, self.s_list.j, self.s_list.shifts)
        e, f = self.student(x)
        return float(e.detach().sum()), f.detach().double()

    def zbl_f(self, x):
        self.z_list.update(x)
        if self.z_list.i is None or self.z_list.i.numel() == 0:
            return 0.0, torch.zeros_like(x)
        e, f = self.zbl(x, self.z_list.i, self.z_list.j, self.z_list.shifts)
        return float(e), f


def adaptive_dt(v: torch.Tensor, xmax: float, dt_min: float, dt_max: float) -> float:
    """The step (fs) in which the fastest atom moves at most xmax (A); v in A / ASE time."""
    # 1 A per ASE time unit is units.fs A per fs (ASE time ~10.18 fs): multiply. Dividing (the first version) read every
    # speed 104x too high and took steps ~100x too short (22,637 steps for 0.3 fs-scale smoke run, 2026-11-13)
    vmax = float(v.norm(dim=1).max()) * units.fs  # A per fs
    return min(dt_max, max(dt_min, xmax / max(vmax, 1e-12)))


def wigner_seitz(x: np.ndarray, sites: np.ndarray, box: np.ndarray, x_start: np.ndarray):
    """Vacant sites and interstitials after removing the centre-of-mass drift from the start (trap 32)."""
    frame = x - (x.mean(0) - x_start.mean(0))
    tree = cKDTree(np.mod(sites, box), boxsize=box)
    _, idx = tree.query(np.mod(frame, box))
    occ = np.bincount(idx, minlength=len(sites))
    return int((occ == 0).sum()), int((occ[occ >= 2] - 1).sum())


def run_cascade(arm: str, x0, v0, masses, border, forces: Forces, args, sites, box, ctrl_args):
    """One cascade in one arm: ``teacher`` (MACE + ZBL every step), ``frost`` (the residual held; the magnitude clause on
    the whole cell, as the first ten-cascade run: an interval of k 1 that fails still stands, there being no smaller one),
    ``frostx`` (the same clause, and a failing k 1 interval runs on the teacher, as the executor's never-lose fallback
    does) or ``frostloc`` (frostx with the clause also applied region by region, frost.locality). Every arm records the damage count and the core temperature every --sample-fs (the sampling time is
    not timed); the held arms record every interval's whole-cell, worst-region and worst-atom staleness."""
    dev = x0.device
    x, v = x0.clone(), v0.clone()
    gen = torch.Generator(device=dev).manual_seed(args.seed_noise)
    t, steps, teacher_calls, rollbacks = 0.0, 0, 0, 0
    gamma = 1.0 / (args.border_tau * units.fs)
    sigma = torch.sqrt(units.kB * args.temperature / masses)
    box_t = torch.as_tensor(box, dtype=x.dtype, device=dev)
    core = ~border.squeeze(1)
    n_core = int(core.sum())
    x_start = x0.cpu().numpy()
    damage, stale_trace = [], []
    clock = {"next": 0.0, "sampling": 0.0}
    counts = {"exact_intervals": 0, "local_refusals": 0, "global_refusals": 0, "over_at_k1": 0}
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    def sample(xx, vv, tt, into):
        while tt >= clock["next"] - 1e-9:
            torch.cuda.synchronize()
            s0 = time.perf_counter()
            vac, inter = wigner_seitz(xx.cpu().numpy(), sites, box, x_start)
            ke = 0.5 * float((masses[core] * vv[core] ** 2).sum())
            into.append([round(tt, 2), vac, inter, round(2.0 * ke / (3.0 * n_core * units.kB), 1)])
            clock["next"] += args.sample_fs
            clock["sampling"] += time.perf_counter() - s0

    def langevin_border(v, dt_ase):
        c1 = math.exp(-gamma * dt_ase)
        noise = torch.randn(v.shape, generator=gen, device=dev, dtype=v.dtype)
        v_new = c1 * v + math.sqrt(1 - c1 * c1) * sigma * noise
        return torch.where(border, v_new, v)

    def step(x, v, f, fast_fn, dt_fs):
        h = dt_fs * units.fs
        v = v + 0.5 * h * f / masses
        x = x + 0.5 * h * v
        v = langevin_border(v, h)
        x = x + 0.5 * h * v
        f_new = fast_fn(x)
        v = v + 0.5 * h * f_new / masses
        return x, v, f_new

    def full(xx):
        return forces.teacher_f(xx)[1] + forces.zbl_f(xx)[1]

    if arm == "teacher":
        f = full(x)
        teacher_calls += 1
        sample(x, v, t, damage)
        while t < args.fs:
            dt = adaptive_dt(v, args.xmax, args.dt_min, args.dt_max)
            x, v, f = step(x, v, f, full, dt)
            t += dt
            steps += 1
            teacher_calls += 1
            sample(x, v, t, damage)
    else:
        local = arm == "frostloc"
        fallback = arm in ("frostx", "frostloc")  # a failing interval of k 1 runs on the teacher
        eps = ctrl_args["epsilon"]
        ctrl = IntervalController(k=ctrl_args["k0"], k_max=ctrl_args["k_max"], epsilon=eps,
                                  period_fs=ctrl_args["period_fs"], dt_fs=args.dt_max, resonance_fraction=0.22)

        def fast(xx):
            return forces.student_f(xx)[1] + forces.zbl_f(xx)[1]

        f_s = fast(x)
        r = forces.teacher_f(x)[1] - forces.student_f(x)[1]
        teacher_calls += 1
        sample(x, v, t, damage)
        exact_next = False
        while t < args.fs:
            k = min(ctrl.k, ctrl.resonance_cap())
            tau = min(k * args.dt_max, args.fs - t)  # the interval, as a duration
            if exact_next:
                # the smallest interval failed a clause: this one runs on the teacher at every inner step, then the held
                # path resumes from a fresh residual (f_s + r is the full force at x, so the switch costs no extra call)
                f = f_s + r
                done = 0.0
                while done < tau - 1e-9:
                    dt = min(adaptive_dt(v, args.xmax, args.dt_min, args.dt_max), tau - done)
                    x, v, f = step(x, v, f, full, dt)
                    done += dt
                    steps += 1
                    teacher_calls += 1
                    sample(x, v, t + done, damage)
                f_s = fast(x)
                r = f - f_s
                stale_trace.append([round(t, 2), k, None, None, None, "x"])
                t += tau
                counts["exact_intervals"] += 1
                exact_next = False
                continue
            ck = (x.clone(), v.clone(), f_s.clone(), gen.get_state(), clock["next"])
            pending = []
            v = v + 0.5 * tau * units.fs * r / masses
            done = 0.0
            while done < tau - 1e-9:
                dt = min(adaptive_dt(v, args.xmax, args.dt_min, args.dt_max), tau - done)
                x, v, f_s = step(x, v, f_s, fast, dt)
                done += dt
                steps += 1
                sample(x, v, t + done, pending)
            r_new = forces.teacher_f(x)[1] - forces.student_f(x)[1]
            teacher_calls += 1
            st = staleness_stats((r_new - r) / masses * 1e3, x, box_t, args.loc_regions, args.loc_z)
            over_global, over_local = st["rms"] > eps, st["local"] > eps
            refused = over_global or (local and over_local)
            verdict = "a" if not refused else ("r" if k > 1 else ("x" if fallback else "o"))
            stale_trace.append([round(t, 2), k, round(st["rms"], 3), round(st["local"], 3), round(st["atom"], 2), verdict])
            if refused and (k > 1 or fallback):
                x, v, f_s = ck[0], ck[1], ck[2]
                gen.set_state(ck[3])
                clock["next"] = ck[4]
                if k > 1:  # roll back and halve, as the magnitude clause does
                    ctrl.k = ctrl.even(max(1, k // 2))
                    rollbacks += 1
                    counts["global_refusals"] += int(over_global)
                    counts["local_refusals"] += int(over_local and not over_global)
                else:
                    exact_next = True
                continue
            if refused:
                counts["over_at_k1"] += 1  # frost: no smaller interval exists, the interval stands
            damage.extend(pending)
            r = r_new
            v = v + 0.5 * tau * units.fs * r / masses
            ctrl.observe((max(st["rms"], st["local"]) if local else st["rms"]) * math.sqrt(3.0), k)
            t += tau
    torch.cuda.synchronize()
    sec = time.perf_counter() - t0 - clock["sampling"]
    vac, inter = wigner_seitz(x.cpu().numpy(), sites, box, x_start)
    peak = max(damage, key=lambda row: row[1]) if damage else [0.0, 0]
    return {"seconds": sec, "ms_per_fs": 1e3 * sec / args.fs, "steps": steps, "teacher_calls": teacher_calls,
            "rollbacks": rollbacks, "vacancies": vac, "interstitials": inter, "peak_vacancies": peak[1],
            "peak_fs": peak[0], "sampling_seconds": clock["sampling"], **counts, "damage_trace": damage,
            "stale_trace": stale_trace}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--element", default="W")
    ap.add_argument("--cells", type=int, default=14, help="BCC cubic cells per side (2 n^3 atoms)")
    ap.add_argument("--student", required=True)
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--pka-ev", type=float, default=500.0)
    ap.add_argument("--fs", type=float, default=6000.0)
    ap.add_argument("--equil-fs", type=float, default=1000.0, help="student-driven thermal equilibration before the PKA")
    ap.add_argument("--xmax", type=float, default=0.05)
    ap.add_argument("--dt-min", type=float, default=0.005)
    ap.add_argument("--dt-max", type=float, default=1.0)
    ap.add_argument("--border-cells", type=float, default=1.5)
    ap.add_argument("--border-tau", type=float, default=100.0, help="border thermostat time constant, fs")
    ap.add_argument("--zbl-in", type=float, default=1.0)
    ap.add_argument("--zbl-out", type=float, default=2.0)
    ap.add_argument("--period-fs", type=float, default=210.5, help="the crystal's fastest period (W 300 K: 210.5 fs)")
    ap.add_argument("--epsilon", type=float, default=2.0)
    ap.add_argument("--k-max", type=int, default=64)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--arms", default="teacher,frost", help="any of teacher, frost, frostx, frostloc (comma-separated)")
    ap.add_argument("--loc-regions", type=int, default=4, help="frostloc: regions per side for the local clause")
    ap.add_argument("--loc-z", type=float, default=4.0, help="frostloc: the regions' allowance, 1 + z / sqrt(6 m)")
    ap.add_argument("--sample-fs", type=float, default=25.0, help="damage count and core temperature every this many fs")
    ap.add_argument("--tight-teacher", action="store_true", help="filter the teacher's skin list to r_max every call")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    dev = "cuda"
    atoms = bcc(args.element, args.cells)
    n = len(atoms)
    box = np.diag(atoms.cell.array).astype(float)
    sites = atoms.get_positions().copy()
    masses = torch.as_tensor(atoms.get_masses(), dtype=torch.float64, device=dev).unsqueeze(1)
    forces = Forces(atoms, args.student, dev, args.zbl_in, args.zbl_out, tight=args.tight_teacher)
    frac = atoms.get_scaled_positions()
    lo, hi = args.border_cells / args.cells, 1 - args.border_cells / args.cells
    border = torch.as_tensor(((frac < lo) | (frac > hi)).any(1), device=dev).unsqueeze(1)
    report = {"element": args.element, "atoms": n, "config": vars(args), "cascades": []}
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        MaxwellBoltzmannDistribution(atoms, temperature_K=2 * args.temperature, rng=rng)
        x = torch.as_tensor(sites, dtype=torch.float64, device=dev)
        v = torch.as_tensor(atoms.get_velocities(), dtype=torch.float64, device=dev)
        # thermal equilibration on the student (cheap; both arms then start from the same state)
        f = forces.student_f(x)[1]
        h = 1.0 * units.fs
        for _ in range(int(args.equil_fs)):
            v = v + 0.5 * h * f / masses
            x = x + h * v
            f = forces.student_f(x)[1]
            v = v + 0.5 * h * f / masses
        # the PKA: the atom nearest the centre, a random direction, net momentum removed
        centre = torch.as_tensor(box / 2, device=dev)
        pka = int((x - centre).norm(dim=1).argmin())
        d = torch.as_tensor(rng.normal(size=3), device=dev, dtype=torch.float64)
        d = d / d.norm()
        speed = math.sqrt(2 * args.pka_ev / float(masses[pka]))  # A / ASE time
        v[pka] = speed * d
        v = v - (masses * v).sum(0) / masses.sum()
        ctrl_args = {"k0": 2, "k_max": args.k_max, "epsilon": args.epsilon, "period_fs": args.period_fs}
        row = {"seed": seed, "pka_atom": pka, "direction": d.tolist(), "arms": {}}
        for arm in args.arms.split(","):
            args.seed_noise = seed
            res = run_cascade(arm, x, v, masses, border, forces, args, sites, box, ctrl_args)
            row["arms"][arm] = res
            held = [s for s in res["stale_trace"] if s[2] is not None]
            worst = max((s[3] / max(s[2], 1e-9) for s in held), default=0.0)
            print(f"seed {seed} {arm}: {res['ms_per_fs']:.1f} ms/fs | steps {res['steps']} | teacher calls {res['teacher_calls']} | "
                  f"rollbacks {res['rollbacks']} | vacancies {res['vacancies']} interstitials {res['interstitials']} | "
                  f"peak {res['peak_vacancies']} at {res['peak_fs']:.0f} fs"
                  + (f" | exact {res['exact_intervals']}, local refusals {res['local_refusals']}, over at k 1 "
                     f"{res['over_at_k1']}, worst region/cell {worst:.2f}" if arm != "teacher" else ""), flush=True)
        report["cascades"].append(row)
        # written after every cascade (atomically), so a job stopped by its time limit keeps the cascades it finished
        OUT.mkdir(parents=True, exist_ok=True)
        out = OUT / f"cascade_{args.element}{args.cells}_{args.tag}.json"
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(report, indent=1))
        tmp.replace(out)
    print("->", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
