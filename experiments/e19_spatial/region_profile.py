"""Where does a region teacher call's time go? (gaps.md, spatial multirate on the thick electrode, 2026-11-15)

On water over 72 layers of Cu(111) (2,898 atoms) an exact region call (the 24 A cut-out, 1,134 atoms, 39%) costs ~56 ms
against a full call's 67, with either gradient path (positions: 57; edge vectors: 56), and a whole cell of similar size
(the 1,170-atom small interface) costs 29 ms. This times, at fixed positions (no list rebuild inside the timed calls):
- the full teacher call (``build``'s graphed teacher, edge forces);
- each region teacher (24:12 and 8:4; positions path and edge path): the whole ``__call__`` (gather, list check, copy,
  replay, output copy) and the graph replay alone;
- a plain graphed whole-cell teacher on the cut-out's own atoms (``GraphedTeacher`` on atoms[sub_idx], same cell), the
  natural floor for a region call;
and records each one's edge count and capacity, then a torch.profiler table of one region call and one full call.

    PYTHONPATH=src python experiments/e19_spatial/region_profile.py --run cuwaterw_T300 \
        --student data/models/students/cuwaterw_T300_f45c16.pt
Output: data/results/e19/region_profile_<run>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "results" / "e19"


def cuda_ms(fn, reps: int = 30) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


def top_kernels(fn, n: int = 12) -> list:
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    rows = []
    def gpu_us(e):  # the attribute's name changed across torch versions
        v = getattr(e, "device_time_total", None)
        return float(v if v is not None else getattr(e, "cuda_time_total", 0.0))

    for e in sorted(prof.key_averages(), key=lambda e: -gpu_us(e))[:n]:
        rows.append({"name": e.key[:90], "cuda_ms": gpu_us(e) / 1e3, "calls": e.count, "cpu_ms": e.cpu_time_total / 1e3})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="cuwaterw_T300")
    ap.add_argument("--student", default="data/models/students/cuwaterw_T300_f45c16.pt")
    ap.add_argument("--buffers", default="24:12,8:4")
    ap.add_argument("--interface-layers", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(4)
    from frost.neighbours import edges_for
    from frost.runtime.md import build
    from frost.spatial import GraphedRegionTeacher
    from frost.teacher_fast import GraphedTeacher
    from frost.teachers import load_calc

    b = build(args.run, args.student, start=9999, dt_fs=1.0, ensemble="nvt", friction=0.001, hmr=3.0,
              edge_forces=True, compile_student="default", derived_lists=False)
    n, atoms = b["n"], b["atoms"]
    dev = b["x0"].device
    numbers = atoms.get_atomic_numbers()
    z = atoms.get_positions()[:, 2]
    cu = numbers == 29
    layers = np.unique(np.round(z[cu], 1))
    z_cut = layers[-args.interface_layers] - 0.3
    fast_idx = np.flatnonzero((~cu) | (cu & (z >= z_cut)))
    calc = load_calc(b["teacher_name"], str(dev), "float32", enable_cueq=True)
    x0 = b["x0"].clone()
    out = {"run": args.run, "n_atoms": n, "gpu": torch.cuda.get_device_name(0), "rows": []}

    def full():
        return b["teacher_force"](x0)

    out["full_ms"] = cuda_ms(full)
    print(f"full teacher call ({n} atoms): {out['full_ms']:.2f} ms", flush=True)
    out["full_kernels"] = top_kernels(full)
    for spec in args.buffers.split(","):
        buf, inner = (float(v) for v in spec.split(":"))
        for edge_path in (False, True):
            g = GraphedRegionTeacher(calc, atoms, fast_idx, buf, inner, str(dev), edge_path=edge_path)
            g(x0)  # builds the list and captures
            row = {"buffer": buf, "inner": inner, "edge_path": edge_path, "sub_atoms": g.n_sub,
                   "edges": int(g.list.i.numel()), "capacity": int(g.capacity), "rebuilds_before": g.list.rebuilds}
            row["call_ms"] = cuda_ms(lambda: g(x0))
            row["replay_ms"] = cuda_ms(lambda: g.graph.replay())
            row["rebuilds_after"] = g.list.rebuilds
            if not edge_path:
                # the floor: a plain graphed whole-cell call on the cut-out's own atoms (same cell, its own edges)
                sub = atoms[g.sub_idx]
                gt = GraphedTeacher(calc, sub, device=str(dev), edge_forces=True)
                xs = torch.as_tensor(sub.get_positions(), dtype=torch.float32, device=dev)
                _, i, j, sh, us = edges_for(sub.get_positions(), np.array(sub.cell), float(calc.models[0].r_max),
                                            device=str(dev), dtype=torch.float32, wrap_positions=False)
                gt.set_edges(i, j, sh, us)
                gt(xs)
                row["plain_sub_ms"] = cuda_ms(lambda: gt(xs))
                row["plain_sub_edges"] = int(i.numel())
                del gt
            row["kernels"] = top_kernels(lambda: g(x0))
            out["rows"].append(row)
            print({k: v for k, v in row.items() if k != "kernels"}, flush=True)
            del g
            torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"region_profile_{args.run}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(path)
    print("->", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
