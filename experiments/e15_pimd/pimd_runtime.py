"""Frost on path-integral MD: multirate in real and imaginary time (further_expansions_beyond.md 6.6).

Arms, each from the same initial ring polymer of P beads (physical masses, no hydrogen repartitioning: quantum effects
depend on the mass):
- ``full``  the teacher on every bead, every step: the reference, P teacher calls per step;
- ``rpc``   ring-polymer contraction with the student as the cheap reference (Markland and Manolopoulos 2008): the student
            on all P beads every step, the residual teacher - student on P' contracted beads every step;
- ``frost`` the same, with Frost's multirate interval: the residual on the contracted ring held for k steps, k adapted by
            the magnitude clause on the contracted residual's change (rollback on failure) under the resonance prior on
            the physical fastest period.
Observables: speed (ms per simulated fs), the centroid-virial quantum kinetic energy per element, and the O-H and H-H
radial distribution functions over all beads. Judged against ``full``.

Usage (GPU, inside a dispatcher job):
  python experiments/e15_pimd/pimd_runtime.py --run water216off_T300 --student data/models/students/<...>.pt \
      --beads 16 --contract 4 --dt 0.5 --fs 4000 --equil 500 --period-fs 9.0 --epsilon-rel 0.043 --tag pimd_s0
Output: data/results/e15/pimd_<run>_<tag>.json
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

from frost.controller import IntervalController
from frost.gpu_guard import ContentionWatch
from frost.neighbours import VerletList, concat_lists, filter_within
from frost.observables import rdf
from frost.pimd import Contraction, RingPolymer, fit_ring_curvature
from frost.student import GraphedStudent
from frost.students import load_student
from frost.teacher_fast import GraphedTeacher
from frost.teachers import load_calc, teacher_of
from frost.teacher import frame_atoms
from frost.trajectory import TrajectoryReader

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "results" / "e15"


class BeadModel:
    """All beads in one graphed call: each bead keeps its own Verlet list, and their edges are concatenated with atom
    offsets b x N into one disconnected graph of B x N atoms. One displacement check covers every bead (one host sync per
    call), only the beads that moved past half the skin rebuild, and the model is called once per step whatever B is.
    The first version called the model bead by bead: 6 calls and 6 syncs per step at P = 4, P' = 2, so the student cost
    7.6 ms per bead against <1 ms for one call (2026-11-13 smoke run)."""

    def __init__(self, graph, set_edges, cutoff: float, skin: float, cell, n_beads: int, n_atoms: int, device: str,
                 tight_cut2: float | None = None):
        self.graph, self.set_edges = graph, set_edges
        # tight (2026-11-17): every call gets the beads' skin lists filtered to the cutoff at the beads' own positions
        # (frost.neighbours.filter_within, one compaction for all beads), as md_runtime --tight-teacher: MACE evaluates
        # every edge it is given, so the skin's pairs cost 21-34% of a call (teacher_list_cost.py)
        self.tight_cut2, self._skin_edges = tight_cut2, None
        self.rebuilds = 0  # bead lists rebuilt (2026-11-17: beads move at P x T, so lists rebuild far more than in MD)
        self.lists = [VerletList(cell, cutoff, skin, device, wrap_positions=False, builder="alchemi")
                      for _ in range(n_beads)]
        self.skin, self.n_beads, self.n = skin, n_beads, n_atoms
        self.x_ref = None
        self.calls = 0  # bead-evaluations (B per call), for cost accounting

    def _sync_edges(self) -> None:
        off = [b * self.n for b in range(self.n_beads)]
        i = torch.cat([lst.i + o for lst, o in zip(self.lists, off, strict=True)])
        j = torch.cat([lst.j + o for lst, o in zip(self.lists, off, strict=True)])
        sh = torch.cat([lst.shifts for lst in self.lists])
        us = torch.cat([lst.unit_shifts for lst in self.lists])
        self.set_edges(i, j, sh, us)

    def __call__(self, q: torch.Tensor):
        if self.x_ref is None:
            stale = list(range(self.n_beads))
        else:
            moved = (q - self.x_ref).norm(dim=-1).amax(dim=-1)  # (B,), one op
            stale = torch.nonzero(moved > 0.5 * self.skin).flatten().tolist()  # one host sync
        if stale:
            for b in stale:
                self.lists[b].update(q[b], force=True)
            self.rebuilds += len(stale)
            self.x_ref = torch.stack([lst.x_ref for lst in self.lists])
            if self.tight_cut2 is None:
                self._sync_edges()
            else:
                self._skin_edges = concat_lists(self.lists, self.n)
        if self.tight_cut2 is not None:
            self.set_edges(*filter_within(*self._skin_edges, q.reshape(-1, 3), self.tight_cut2))
        e, f = self.graph(q.reshape(-1, 3))
        self.calls += self.n_beads
        return torch.as_tensor(e).detach().reshape(()).double().expand(self.n_beads), f.detach().double().view(q.shape)


def initial_ring(x0: torch.Tensor, masses: torch.Tensor, p: int, temperature: float, seed: int):
    """Every bead at the classical positions x0, velocities at P T (as i-PI starts): the springs spread the ring within a
    few fs while the potential keeps bonds physical. Drawing beads from the *free* ring polymer instead (~0.2 A per
    hydrogen bead, blind to the O-H bond) squeezed some bonds and gave 1,205 eV/A on the first call (2026-11-13)."""
    from frost.pimd import NormalModes

    g = torch.Generator(device=x0.device).manual_seed(seed)
    modes = NormalModes(p, temperature, device=x0.device)
    q = x0.unsqueeze(0).repeat(p, 1, 1)
    v = torch.randn((p, *x0.shape), generator=g, device=x0.device, dtype=torch.float64) * torch.sqrt(
        modes.kT_P / masses).unsqueeze(0)
    return q, v


def staleness(r_new: torch.Tensor, r_old: torch.Tensor, masses: torch.Tensor) -> float:
    """RMS change of the held residual's acceleration over the contracted beads, meV/(A amu) per component."""
    d = (r_new - r_old) / masses.unsqueeze(0)
    return float(d.pow(2).mean().sqrt()) * 1e3


def observe(ring: RingPolymer, f: torch.Tensor, numbers: np.ndarray, cell, acc: dict) -> None:
    dq = ring.q - ring.centroid().unsqueeze(0)
    k_atom = 1.5 * units.kB * ring.temperature - 0.5 / ring.P * (dq * f).sum((0, 2))  # per atom, eV
    for z in np.unique(numbers):
        sel = torch.as_tensor(numbers == z, device=f.device)
        acc.setdefault(f"K_{int(z)}", []).append(float(k_atom[sel].mean()))
    # every atom's own estimator, summed on the GPU (2026-11-17): a per-element mean cannot say whether a shift sits on
    # a solute's few atoms or on the solvent (the peptide's 12 H among 920)
    acc["k_atom_sum"] = acc.get("k_atom_sum", 0.0) + k_atom.detach().double()
    acc["k_atom_n"] = acc.get("k_atom_n", 0) + 1
    acc.setdefault("frames", []).append(ring.q.detach().float().cpu())


def run_arm(name: str, k_fixed: int, ring: RingPolymer, student: BeadModel, student_c: BeadModel, teacher: BeadModel,
            teacher_c: BeadModel, contraction: Contraction, masses, steps: int, equil: int, epsilon: float,
            ctrl: IntervalController | None, numbers, cell, every: int, rollback_policy: str = "halve",
            ring_hold: int = 0):
    acc = {}
    for model in (teacher, teacher_c, student, student_c):
        if model is not None:
            model.calls = 0
            model.rebuilds = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    events = {"rollbacks": 0, "intervals": 0, "k_sum": 0}
    if name == "full":
        _, f = teacher(ring.q)
        for step in range(steps):
            _, f = ring.step(teacher, f)
            if step >= equil and step % every == 0:
                observe(ring, f, numbers, cell, acc)
    else:
        # ring hold (2026-11-17): the residual's imaginary-time structure, held the way its real-time value is. With P'
        # 1 every bead gets the centroid's residual, so the ring's spread feels the student's curvature (water H read
        # its quantum kinetic energy -0.24% low in the peptide's solvent, -0.47% at P' 1 before ring-pair training; gone
        # at P' 16). The residual's curvature at each atom changes slowly (it follows the molecule's geometry) while the
        # beads' offsets from their centroid change every step, so every ``ring_hold`` steps the teacher and the student
        # run on all P beads, each atom's 3x3 J_i is fitted to the residual's bead-to-bead variation (dR_b = J_i dq_b,
        # least squares over the beads, symmetrised so the extra force is conservative), and every step each bead gets
        # J_i (q_b - q_centroid): zero summed over the beads, so the centroid's residual is untouched.
        hold = {"J": None, "since": 0, "explained": []}

        def refresh_curvature():
            q = ring.q
            dq = q - q.mean(0, keepdim=True)
            dr = teacher(q)[1] - student(q)[1]
            j, explained = fit_ring_curvature(dr - dr.mean(0, keepdim=True), dq)
            hold["J"], hold["since"] = j, 0
            hold["explained"].append(explained)

        def force_fn(q):
            e, f = student(q)
            if hold["J"] is not None:
                f = f + torch.einsum("nij,bnj->bni", hold["J"], q - q.mean(0, keepdim=True))
            return e, f

        if ring_hold:
            refresh_curvature()
        _, f_s = force_fn(ring.q)
        q_c = contraction.contract(ring.q)
        r_c = teacher_c(q_c)[1] - student_c(q_c)[1]
        r_full = contraction.expand_force(r_c)
        done = 0
        pending: list[int] = []  # "halves": the two halves of a rolled-back interval, run before the controller's next k
        trace = []  # (k, staleness, rolled back) per attempted interval of an adaptive arm
        while done < steps:
            k = pending.pop(0) if pending else (k_fixed or ctrl.k)
            k = max(1, min(k, steps - done))
            ck = (ring.q.clone(), ring.v.clone(), f_s.clone(), ring.pile.gen.get_state())
            ring.kick(r_full, 0.5 * k * ring.dt)
            for _ in range(k):
                _, f_s = ring.step(force_fn, f_s)
            q_c = contraction.contract(ring.q)
            r_c_new = teacher_c(q_c)[1] - student_c(q_c)[1]
            stale = staleness(r_c_new, r_c, masses)
            failed = not k_fixed and stale > epsilon and k > 1
            if not k_fixed:
                trace.append((k, round(float(stale), 4), bool(failed)))
            if failed:
                ring.q, ring.v, f_s = ck[0], ck[1], ck[2]
                ring.pile.gen.set_state(ck[3])
                events["rollbacks"] += 1
                if rollback_policy == "halves":
                    # the unified runtime's policy (runtime/executor.py, Executor._one): redo the refused interval as
                    # two halves, each judged on its own, and leave the controller on its rung; "halve" (the first
                    # version) dropped the controller to k // 2, so a refusal at k 3 sent it to k 1 and it climbed back
                    pending[:0] = [k // 2, k - k // 2]
                else:
                    ctrl.k = ctrl.even(max(1, k // 2))
                continue
            r_c = r_c_new
            r_full = contraction.expand_force(r_c)
            ring.kick(r_full, 0.5 * k * ring.dt)
            if not k_fixed:
                ctrl.observe(stale * math.sqrt(3.0), k)  # the controller takes the per-atom change (MDBackend's convention)
            events["intervals"] += 1
            events["k_sum"] += k
            if ring_hold:
                hold["since"] += k
                if hold["since"] >= ring_hold:
                    refresh_curvature()
                    _, f_s = force_fn(ring.q)  # the next step's first half-kick sees the new curvature
            for s in range(done, done + k):  # observe on the closed interval's last state, at the sampling period
                if s >= equil and s % every == 0:
                    observe(ring, f_s + r_full, numbers, cell, acc)
                    break
            done += k
    torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    out = {"ms_per_fs": 1e3 * sec / (steps * ring.dt / units.fs),
           "teacher_calls": (teacher.calls if teacher is not None else 0) + teacher_c.calls,
           "student_calls": student.calls + student_c.calls,
           # bead-list rebuilds per step, per model (each rebuild: a cell-list build and an edge write)
           "rebuilds_per_step": {nm: m.rebuilds / steps for nm, m in (("teacher", teacher), ("teacher_c", teacher_c),
                                                                      ("student", student), ("student_c", student_c))
                                 if m is not None}}
    if events["intervals"]:
        out.update(mean_k=events["k_sum"] / events["intervals"], rollbacks=events["rollbacks"])
    if name != "full" and ring_hold:
        ex = hold["explained"]
        out["ring_hold"] = {"every_steps": ring_hold, "refreshes": len(ex),
                            "explained_mean": float(np.mean(ex)) if ex else None,
                            "explained_min": float(np.min(ex)) if ex else None}
    if name != "full" and not k_fixed:
        out["trace"] = trace
        out["rollback_policy"] = rollback_policy
    for key, vals in acc.items():
        if key.startswith("K_"):
            out[key + "_meV"] = 1e3 * float(np.mean(vals))
    if acc.get("k_atom_n"):
        # not "K_..." (every "K_" key is a per-element scalar: printed, and divided by the full arm's for a ratio)
        out["atomK_meV"] = (1e3 * acc["k_atom_sum"] / acc["k_atom_n"]).cpu().numpy().round(4).tolist()
    if acc.get("frames"):
        x = torch.cat(acc["frames"], 0).double()  # every bead of every sampled frame
        species, types = np.unique(numbers, return_inverse=True)
        g = rdf(x, cell, types, len(species))
        out["rdf"] = {k: (np.asarray(v).tolist() if hasattr(v, "__len__") else v) for k, v in g.items()}
    return out


def dump_beads(args, teacher, x0, masses, numbers, cell, teacher_name, temperature) -> int:
    """The bead paths of the teacher's own PIMD as a training trajectory (gaps.md, PIMD loss, 2026-11-14).

    A student trained on classical frames never sees a ring-polymer bead: with real hydrogen masses each H bead sits
    ~0.2 A off the classical position, so the held residual there was large and fast and Frost stayed at k 1. Each bead's
    path is a continuous trajectory in time, so ``replay.py`` can train on it unchanged; ``--dump-beads`` evenly spaced
    beads are written one after another (the few junctions between them are negligible against the frames)."""
    from frost.trajectory import TrajectoryWriter

    p = args.beads
    pick = [b * p // args.dump_beads for b in range(args.dump_beads)]
    ring = RingPolymer(*initial_ring(x0, masses, p, temperature, args.seed), masses, temperature, args.dt, args.tau0,
                       args.seed)
    equil, steps = round(args.equil / args.dt), round(args.dump_fs / args.dt)
    _, f = teacher(ring.q)
    for _ in range(equil):
        _, f = ring.step(teacher, f)
    paths = [[] for _ in pick]
    energies = [[] for _ in pick]
    for _ in range(steps):
        e, f = ring.step(teacher, f)
        for slot, b in enumerate(pick):
            paths[slot].append(ring.q[b].detach().cpu().numpy().copy())
            energies[slot].append(float(e[b]) if hasattr(e, "__len__") else float("nan"))
    label = args.dump_label or f"{args.run.split('_T')[0]}pimd"
    out = ROOT / "data" / "trajectories" / f"{label}_T{int(temperature)}.h5"
    atoms = frame_atoms(numbers, x0.cpu().numpy(), cell)
    attrs = {"cell": np.asarray(cell), "numbers": np.asarray(numbers).astype(np.uint8), "pbc": np.array(atoms.pbc),
             "temperature_K": temperature, "timestep_fs": args.dt, "cutoff": 6.0, "skin": 1.0, "seed": args.seed,
             "teacher": teacher_name, "generator": "pimd_runtime.py --dump-beads",
             "source": f"{args.run} PIMD P {p}, beads {pick}"}
    zeros = np.zeros((len(numbers), 3), dtype=np.float32)
    step = 0
    with TrajectoryWriter(out, n_atoms=len(numbers), cutoff=6.0, skin=1.0, attrs=attrs) as writer:
        for slot in range(len(pick)):
            for x_np, e in zip(paths[slot], energies[slot], strict=True):
                atoms.set_positions(x_np)
                atoms.wrap()
                writer.write(step, x_np, atoms, e, temperature, zeros)  # bead velocities are not physical velocities
                step += 1
    print(f"-> {out}: beads {pick}, {steps} steps each, {step} frames", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--beads", type=int, default=16)
    ap.add_argument("--contract", type=int, default=4)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--fs", type=float, default=4000.0)
    ap.add_argument("--equil", type=float, default=500.0, help="fs discarded before observing, per arm")
    ap.add_argument("--start", type=int, default=9000)
    ap.add_argument("--period-fs", type=float, default=9.0, help="the physical fastest period (O-H stretch ~9 fs)")
    ap.add_argument("--resonance-fraction", type=float, default=0.22)
    ap.add_argument("--k-max", type=int, default=16)
    ap.add_argument("--epsilon-rel", type=float, default=0.043)
    ap.add_argument("--arms", default="full,rpc,frost")
    ap.add_argument("--fixed-k", default="", help="extra fixed-k contracted arms, e.g. 2,4")
    ap.add_argument("--tau0", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=20, help="steps between observations")
    ap.add_argument("--dump-beads", type=int, default=0,
                    help="instead of the arms: run the teacher's full PIMD and write this many evenly spaced beads' "
                         "paths, every step, as one trajectory a student trains on (replay.py)")
    ap.add_argument("--dump-fs", type=float, default=1500.0, help="fs written per bead (after --equil)")
    ap.add_argument("--dump-label", default="", help="trajectory label: data/trajectories/<label>_T<T>.h5")
    ap.add_argument("--ring-hold", type=int, default=0,
                    help="steps between full-ring refreshes of each atom's held residual curvature (0: off); needs the "
                         "teacher on all P beads, built even without a full arm")
    ap.add_argument("--rollback-policy", choices=("halve", "halves"), default="halve",
                    help="after a refused interval: halve the controller's k (first version), or redo the interval as "
                         "two halves and keep the controller's rung (the unified runtime's executor)")
    ap.add_argument("--tight-student", action="store_true",
                    help="the student's bead lists filtered to its cutoff before every call, its graph sized from the "
                         "filtered list: the skin then sets only how often lists rebuild, not what a call computes")
    ap.add_argument("--student-skin", type=float, default=0.5, help="A: the student's Verlet skin")
    ap.add_argument("--teacher-skin", type=float, default=1.0,
                    help="A: the teacher's Verlet skin (with --tight-teacher it sets only how often lists rebuild)")
    ap.add_argument("--tight-teacher", action="store_true",
                    help="filter the teacher's skin lists to r_max before every call (the strongest baseline, 2026-11-17)")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    dev = "cuda"
    path = ROOT / "data" / "trajectories" / f"{args.run}.h5"
    with TrajectoryReader(path) as rd:
        numbers, cell = rd.numbers.copy(), rd.cell.copy()
        x0 = torch.as_tensor(np.asarray(rd.positions[args.start]), dtype=torch.float64, device=dev)
        teacher_name = teacher_of(rd.attrs)
    temperature = float(args.run.split("_T")[1])
    atoms = frame_atoms(numbers, x0.cpu().numpy(), cell)
    masses = torch.as_tensor(atoms.get_masses(), dtype=torch.float64, device=dev).unsqueeze(1)
    calc = load_calc(teacher_name, dev, "float32", enable_cueq=True)
    st = load_student(args.student, dev)
    if st.config.get("kind", "frost") == "mace" and st.config.get("backend", "e3nn") != "cueq":
        st = st.converted("cueq")
    types1 = st.types(torch.as_tensor(numbers, device=dev))
    s_cut = float(st.config["cutoff"])
    lst1 = VerletList(cell, s_cut, args.student_skin, dev, wrap_positions=False, builder="alchemi")
    lst1.update(x0, force=True)
    s_cut2 = s_cut ** 2 if args.tight_student else None
    s_edges1 = (filter_within(lst1.i, lst1.j, lst1.shifts, lst1.unit_shifts, x0, s_cut2) if s_cut2
                else (lst1.i, lst1.j, lst1.shifts, lst1.unit_shifts))
    per_bead_edges = int(s_edges1[0].numel())
    p, pc = args.beads, args.contract
    n = len(numbers)

    def teacher_for(nb: int):
        """A teacher graph over nb copies of the system (a disconnected graph of nb x N atoms)."""
        ring_atoms = frame_atoms(np.tile(numbers, nb), np.tile(x0.cpu().numpy(), (nb, 1)), cell)
        g = GraphedTeacher(calc, ring_atoms, dev, edge_forces=True)
        lists = [VerletList(cell, g.r_max, args.teacher_skin, dev, wrap_positions=False, builder="alchemi")
                 for _ in range(nb)]
        for lst in lists:
            lst.update(x0, force=True)
        cut2 = float(g.r_max) ** 2 if args.tight_teacher else None
        edges = concat_lists(lists, n)
        # tight: the graph's buffers are sized from the filtered list (they only grow)
        g.set_edges(*(filter_within(*edges, x0.repeat(nb, 1), cut2) if cut2 else edges))
        model = BeadModel(g, lambda i, j, sh, us: g.set_edges(i, j, sh, us), g.r_max, args.teacher_skin, cell, nb, n,
                          dev, tight_cut2=cut2)
        return model, g

    def student_for(nb: int):
        # tight: capacity 1.25x the filtered list (it grows by recapture if a call needs more); a padded slot costs as
        # much as an edge, so a capacity sized from the skin list would keep the skin's cost
        cap = int((1.25 if s_cut2 else 1.5) * nb * per_bead_edges) + 64
        g = GraphedStudent(st, types1.repeat(nb), nb * n, capacity=cap)
        g.set_edges(torch.cat([s_edges1[0] + b * n for b in range(nb)]),
                    torch.cat([s_edges1[1] + b * n for b in range(nb)]), torch.cat([s_edges1[2]] * nb))
        return BeadModel(g, lambda i, j, sh, us: g.set_edges(i, j, sh), s_cut, args.student_skin, cell, nb, n, dev,
                         tight_cut2=s_cut2)

    if args.dump_beads:
        return dump_beads(args, teacher_for(p)[0], x0, masses, numbers, cell, teacher_name, temperature)
    teacher, tg_full = teacher_for(p) if ("full" in args.arms.split(",") or args.ring_hold) else (None, None)
    teacher_c, tg_c = teacher_for(pc)
    student, student_c = student_for(p), student_for(pc)
    contraction = Contraction(p, pc, device=dev)
    f0 = teacher_c(x0.unsqueeze(0).repeat(pc, 1, 1))[1][0]
    if args.tight_teacher:
        # self-check: the tight graphed teacher against the calculator's own eager call (its own neighbour list)
        f_ref = torch.as_tensor(calc.get_forces(frame_atoms(numbers, x0.cpu().numpy(), cell)), dtype=torch.float64,
                                device=dev)
        d_f = float((f0 - f_ref).abs().max())
        print(f"self-check: tight graphed teacher against the eager call, max |dF| {d_f:.2e} of max |F| "
              f"{float(f_ref.abs().max()):.2f} eV/A", flush=True)
        if d_f > 1e-3:
            raise SystemExit("the tight teacher's forces differ from the eager call")
    epsilon = args.epsilon_rel * float(1e3 * ((f0 / masses) ** 2).mean().sqrt())
    print(f"{args.run}: {len(numbers)} atoms, P {p} -> P' {pc}, teacher {teacher_name}, epsilon {epsilon:.3f}", flush=True)
    q0, v0 = initial_ring(x0, masses, p, temperature, args.seed)
    steps = round(args.fs / args.dt)
    equil = round(args.equil / args.dt)
    report = {"run": args.run, "config": vars(args), "epsilon": epsilon, "arms": {}}
    # an --arms entry "fixedN" is a fixed-k arm at that position (2026-11-17: arm order is part of a timing, so it can be
    # set; --fixed-k arms run after the --arms ones)
    arms = [(a, int(a[5:]) if a.startswith("fixed") else 0) for a in args.arms.split(",") if a] + \
        [(f"fixed{k}", int(k)) for k in args.fixed_k.split(",") if k]
    for name, kf in arms:
        ring = RingPolymer(q0.clone(), v0.clone(), masses, temperature, args.dt, args.tau0, args.seed)
        ctrl = None
        if name == "frost":
            ctrl = IntervalController(k=2, k_max=args.k_max, epsilon=epsilon, period_fs=args.period_fs, dt_fs=args.dt,
                                      resonance_fraction=args.resonance_fraction, fine_fs=4.0)
        k_fixed = {"rpc": 1}.get(name, kf)
        watch = ContentionWatch().start()  # was the arm alone on its GPU? (frost.gpu_guard; NVML, outside the stream)
        res = run_arm(name, k_fixed, ring, student, student_c, teacher, teacher_c, contraction, masses, steps, equil,
                      epsilon, ctrl, numbers, cell, args.every, args.rollback_policy, args.ring_hold)
        res["contention"] = watch.stop()
        if res["contention"].get("contended"):
            # 2026-11-17: another tenant returned mid-run and slowed fixed 3 by 17% against the certified arm doing the
            # same work at k 3.0; without this record the arm was indistinguishable from a clean one
            print(f"{name} arm SHARED its GPU (foreign active {res['contention'].get('foreign_active_fraction')} of "
                  f"samples): its timing is not quotable", flush=True)
        report["arms"][name] = res
        print(f"{name}: {res['ms_per_fs']:.1f} ms/fs | mean k {res.get('mean_k', 1):.1f} | rollbacks {res.get('rollbacks', 0)} | "
              + "rebuilds/step " + " ".join(f"{m} {v:.2f}" for m, v in res["rebuilds_per_step"].items()) + " | "
              + " ".join(f"{k} {v:.1f}" for k, v in res.items() if k.startswith("K_")), flush=True)
    ref = report["arms"].get("full")
    if ref:
        for name, res in report["arms"].items():
            res["speedup_vs_full"] = ref["ms_per_fs"] / res["ms_per_fs"]
            for key in [k for k in res if k.startswith("K_")]:
                res[key.replace("_meV", "_ratio")] = res[key] / ref[key]
            if "rdf" in res and "rdf" in ref:
                a, b = np.asarray(ref["rdf"]["g_total"]), np.asarray(res["rdf"]["g_total"])
                res["rdf_l1_vs_full"] = float(np.abs(b - a).sum() / np.abs(a).sum())
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"pimd_{args.run}_{args.tag}.json"
    out.write_text(json.dumps(report, indent=1, default=float))
    print("->", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
