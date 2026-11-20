"""Where a path-integral step's time goes, on the tight-teacher baseline (2026-11-17).

The tight teacher (every call on the skin list filtered to r_max) took the full PIMD arm from 209.8 to 137.3 ms/fs, while
ring-polymer contraction with the MACE student moved only from 147.1 to 133.6: the student on all 16 beads, which every
Frost arm pays every step, now costs about as much as the teacher on them. This measures each piece of a step on one GPU,
with the runner's own objects (``pimd_runtime.BeadModel``, ``RingPolymer``, ``Contraction``), wall time over repeated
calls after a synchronize (a BeadModel call includes its displacement check, a host sync):
- the teacher on P and on P' beads, skin list and tight;
- each student on P beads, as the runner builds it (skin list, capacity 1.5x) and tight (capacity from the tight list);
- the ring-polymer step without a force call, and the contraction and expansion;
and composes the per-step cost of each arm (full, rpc, Frost at k) from those pieces, so a claim about which piece to
change rests on its measured share.

    python experiments/e15_pimd/pimd_profile.py [--run water216off_T300] [--students m45c16L,f45c16,...] [--beads 16]
Output: data/results/e15/pimd_profile_<run>.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "results" / "e15"


def _runner():
    spec = importlib.util.spec_from_file_location("pimd_runtime", Path(__file__).with_name("pimd_runtime.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def wall_ms(fn, reps: int = 40) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return 1e3 * (time.perf_counter() - t0) / reps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="water216off_T300")
    ap.add_argument("--students", default="water216off_T300_m45c16L,water216off_T300_f45c16,water216offpimd_T300_f45c16L")
    ap.add_argument("--beads", type=int, default=16)
    ap.add_argument("--contract", type=int, default=2)
    ap.add_argument("--start", type=int, default=9000)
    ap.add_argument("--dt", type=float, default=0.5)
    args = ap.parse_args()
    rt = _runner()
    from frost.neighbours import VerletList, concat_lists, filter_within
    from frost.pimd import Contraction, RingPolymer
    from frost.student import GraphedStudent
    from frost.students import load_student
    from frost.teacher import frame_atoms
    from frost.teacher_fast import GraphedTeacher
    from frost.teachers import load_calc, teacher_of
    from frost.trajectory import TrajectoryReader

    dev = "cuda"
    with TrajectoryReader(ROOT / "data" / "trajectories" / f"{args.run}.h5") as rd:
        numbers, cell = rd.numbers.copy(), rd.cell.copy()
        x0 = torch.as_tensor(np.asarray(rd.positions[args.start]), dtype=torch.float64, device=dev)
        teacher_name = teacher_of(rd.attrs)
    n, p, pc = len(numbers), args.beads, args.contract
    atoms = frame_atoms(numbers, x0.cpu().numpy(), cell)
    masses = torch.as_tensor(atoms.get_masses(), dtype=torch.float64, device=dev).unsqueeze(1)
    temperature = float(args.run.split("_T")[1])
    q0, v0 = rt.initial_ring(x0, masses, p, temperature, 0)
    g0 = torch.Generator(device=dev).manual_seed(1)
    q0 = q0 + 0.03 * torch.randn(q0.shape, generator=g0, device=dev, dtype=q0.dtype)  # distinct beads
    calc = load_calc(teacher_name, dev, "float32", enable_cueq=True)
    out = {"run": args.run, "atoms": n, "beads": p, "contract": pc, "teacher": teacher_name,
           "gpu": torch.cuda.get_device_name(0), "when": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}
    contraction = Contraction(p, pc, device=dev)
    q_c = contraction.contract(q0)

    def teacher_model(nb: int, tight: bool):
        ring_atoms = frame_atoms(np.tile(numbers, nb), np.tile(x0.cpu().numpy(), (nb, 1)), cell)
        g = GraphedTeacher(calc, ring_atoms, dev, edge_forces=True)
        lists = [VerletList(cell, g.r_max, 1.0, dev, wrap_positions=False, builder="alchemi") for _ in range(nb)]
        qq = q0 if nb == p else q_c
        for b, lst in enumerate(lists):
            lst.update(qq[b], force=True)
        cut2 = float(g.r_max) ** 2 if tight else None
        edges = concat_lists(lists, n)
        g.set_edges(*(filter_within(*edges, qq.reshape(-1, 3), cut2) if tight else edges))
        m = rt.BeadModel(g, lambda i, j, sh, us: g.set_edges(i, j, sh, us), g.r_max, 1.0, cell, nb, n, dev,
                         tight_cut2=cut2)
        return m, g

    rows = {}
    for nb, qq in ((p, q0), (pc, q_c)):
        for tight in (False, True):
            m, g = teacher_model(nb, tight)
            rows[f"teacher_P{nb}_{'tight' if tight else 'skin'}"] = {
                "ms": wall_ms(lambda: m(qq)), "edge_capacity": int(g.capacity)}
            del m, g
            torch.cuda.empty_cache()
    for label in args.students.split(","):
        path = ROOT / "data" / "models" / "students" / f"{label}.pt"
        if not path.exists():
            rows[f"student_{label}"] = {"missing": True}
            continue
        st = load_student(str(path), dev)
        if st.config.get("kind", "frost") == "mace" and st.config.get("backend", "e3nn") != "cueq":
            st = st.converted("cueq")
        types1 = st.types(torch.as_tensor(numbers, device=dev))
        s_cut = float(st.config["cutoff"])
        for nb, qq in ((p, q0), (pc, q_c)):
            lists = [VerletList(cell, s_cut, 0.5, dev, wrap_positions=False, builder="alchemi") for _ in range(nb)]
            for b, lst in enumerate(lists):
                lst.update(qq[b], force=True)
            skin_edges = concat_lists(lists, n)
            tight_edges = filter_within(*skin_edges, qq.reshape(-1, 3), s_cut ** 2)
            for tight in (False, True):
                e = tight_edges if tight else skin_edges
                cap = int(1.25 * e[0].numel()) + 64 if tight else int(1.5 * skin_edges[0].numel()) + 64
                g = GraphedStudent(st, types1.repeat(nb), nb * n, capacity=cap)
                g.set_edges(e[0], e[1], e[2])
                m = rt.BeadModel(g, lambda i, j, sh, us, g=g: g.set_edges(i, j, sh), s_cut, 0.5, cell, nb, n, dev,
                                 tight_cut2=s_cut ** 2 if tight else None)
                rows[f"student_{label}_P{nb}_{'tight' if tight else 'skin'}"] = {
                    "ms": wall_ms(lambda: m(qq)), "edges": int(e[0].numel()), "capacity": int(g.capacity),
                    "kind": st.config.get("kind", "frost")}
                del m, g
                torch.cuda.empty_cache()
    # the ring-polymer step without a force call, and the contraction
    zero = torch.zeros_like(q0)

    def no_force(q):
        return torch.zeros(p, dtype=torch.float64, device=dev), zero

    ring = RingPolymer(q0.clone(), v0.clone(), masses, temperature, args.dt, 100.0, 0)
    state = {"f": zero}

    def ring_step():
        state["f"] = ring.step(no_force, state["f"])[1]

    rows["ring_step_no_force"] = {"ms": wall_ms(ring_step, reps=200)}
    rows["contract_expand"] = {"ms": wall_ms(lambda: contraction.expand_force(contraction.contract(q0)), reps=200)}
    out["pieces"] = rows
    # per-step cost of each arm from the pieces (ms per step; x 2 for ms/fs at 0.5 fs)
    ring_ms, ce = rows["ring_step_no_force"]["ms"], rows["contract_expand"]["ms"]
    arms = {}
    for tt in ("skin", "tight"):
        arms[f"full_teacher_{tt}"] = rows[f"teacher_P{p}_{tt}"]["ms"] + ring_ms
    for label in args.students.split(","):
        for st_t in ("skin", "tight"):
            key = f"student_{label}_P{p}_{st_t}"
            if key not in rows:
                continue
            s16, s2 = rows[key]["ms"], rows[f"student_{label}_P{pc}_{st_t}"]["ms"]
            t2 = rows[f"teacher_P{pc}_tight"]["ms"]
            for k in (1, 2, 3, 4, 8):
                arms[f"{label}_{st_t}_k{k}"] = s16 + ring_ms + (t2 + s2 + ce) / k
    out["per_step_ms"] = arms
    out["speedup_vs_full_tight"] = {k: arms["full_teacher_tight"] / v for k, v in arms.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"pimd_profile_{args.run}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(path)
    for k, v in rows.items():
        print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()}, flush=True)
    for k, v in arms.items():
        print(f"{k}: {v:.2f} ms/step -> {out['speedup_vs_full_tight'][k]:.2f}x of the tight full arm", flush=True)
    print("->", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
