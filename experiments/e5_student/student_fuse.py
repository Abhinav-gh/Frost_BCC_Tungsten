"""Is the student kernel-count bound, and does fusion fix it? (MD e2e bottleneck)

On liquid Cu the student is a 2-layer, 16-channel, l <= 1 MACE (~7k edges for 500
atoms), yet one force call costs ~2.8 ms inside a CUDA graph. That is several
times the teacher's per-atom FLOP rate, so the call is made of hundreds of tiny
kernels, each a few microseconds of device time. A CUDA graph removes launch cost,
not kernel count. Fusing kernels is what removes kernel count.

This times one force call (energy forward, then autograd for -dE/dx) per backend:
- **graph**: the call as ``GraphedStudent`` captures it (the production path);
- **compiled + graph**: the same module through ``torch.compile`` (Inductor fuses
  elementwise and reduction chains; AOTAutograd compiles the backward), then
  captured in our own CUDA graph.

It also reports the largest force difference against the uncompiled call, so a
faster path is only adopted if it is the same function. Public APIs only:
``torch.compile`` and ``torch.cuda.graph``; the model is untouched.

Output: data/results/e5/student_fuse_<run>.json

Usage (guarded launcher; GPU)
  scripts/safe_run.sh student_fuse .venvs/frost-mace/bin/python \\
      experiments/e5_student/student_fuse.py --run cu500_T1600 \\
      --student data/models/students/cu500_T1600_s35c16e1liq.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def time_graph(fn, reps: int = 50) -> float:
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    g.replay()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(reps):
            g.replay()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / reps)
    return 1e3 * best, out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="cu500_T1600")
    ap.add_argument("--student", default="data/models/students/cu500_T1600_s35c16e1liq.pt")
    ap.add_argument("--backends", default="cueq,e3nn,oeq")
    ap.add_argument("--skin", type=float, default=0.3)
    args = ap.parse_args()

    from frost.neighbours import VerletList
    from frost.students import load_student

    dev = "cuda"
    with h5py.File(ROOT / "data" / "trajectories" / f"{args.run}.h5", "r") as fh:
        numbers = np.asarray(fh.attrs["numbers"], dtype=np.int64)
        cell = np.asarray(fh.attrs["cell"], dtype=np.float64)
        pos = np.asarray(fh["positions"][5000], dtype=np.float64)
    base = load_student(ROOT / args.student, dev)
    types = base.types(torch.as_tensor(numbers, device=dev))
    nl = VerletList(cell, float(base.config["cutoff"]), args.skin, dev, wrap_positions=False,
                    builder="alchemi")
    x64 = torch.as_tensor(pos, dtype=torch.float64, device=dev)
    nl.update(x64, force=True)
    cap = int(1.3 * nl.i.numel()) + 64
    n_e = int(nl.i.numel())
    i = torch.zeros(cap, dtype=torch.long, device=dev)
    j = torch.zeros(cap, dtype=torch.long, device=dev)
    sh = torch.zeros(cap, 3, device=dev)
    mask = torch.zeros(cap, device=dev)
    i[:n_e], j[:n_e], sh[:n_e], mask[:n_e] = nl.i, nl.j, nl.shifts.float(), 1.0
    x = x64.float().clone().requires_grad_(True)

    rows, reference = {}, None
    for backend in args.backends.split(","):
        try:
            model = base if base.config.get("backend", "e3nn") == backend else base.converted(backend)
        except Exception as exc:  # a backend that is not installed is reported, not fatal
            rows[backend] = {"error": repr(exc)[:200]}
            print(f"{backend}: unavailable ({exc!r:.120})", flush=True)
            continue

        def call(m=model):
            e = m(x, types, i, j, sh, edge_mask=mask)
            (g,) = torch.autograd.grad(e.sum(), x)
            return -g

        ms_graph, f_graph = time_graph(call)
        f_graph = f_graph.clone()
        if reference is None:
            reference = f_graph
        r = {"graph_ms": ms_graph, "force_diff_vs_first": float((f_graph - reference).abs().max())}
        try:
            compiled = torch.compile(model, mode="max-autotune-no-cudagraphs", dynamic=False)
            t0 = time.perf_counter()

            def ccall(m=compiled):
                e = m(x, types, i, j, sh, edge_mask=mask)
                (g,) = torch.autograd.grad(e.sum(), x)
                return -g

            ccall()  # compile outside the capture
            torch.cuda.synchronize()
            r["compile_s"] = time.perf_counter() - t0
            ms_c, f_c = time_graph(ccall)
            r["compiled_graph_ms"] = ms_c
            r["compiled_force_diff"] = float((f_c - f_graph).abs().max())
        except Exception as exc:
            r["compile_error"] = repr(exc)[:300]
        rows[backend] = r
        print(f"{backend}: {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})}",
              flush=True)
    out = ROOT / "data" / "results" / "e5" / f"student_fuse_{args.run}.json"
    out.write_text(json.dumps({"atoms": len(numbers), "edges": n_e, "config": base.config, "rows": rows},
                              indent=1))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
