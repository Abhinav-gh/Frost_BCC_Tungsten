"""C13: how the CFD / MHD speedups depend on the ensemble size (timing only; the statistics are
batch-independent and come from ``stats3d.py``).

Each row is timed at batch 1, 4, 16 and 64 (ms per 0.04 per trajectory), with the same compiled
functions ``stats3d.py`` times: ``classical_advance`` for classical rows, ``split3d.advance`` for
learned ones (``:h`` = bfloat16 network).

Output: data/results/e10/batch_sweep_<tag>.json

Usage (guarded launcher; GPU, JAX env)
  scripts/safe_run.sh cfd_batch env -u LD_LIBRARY_PATH -u CUDA_HOME PYTHONPATH=src \\
      .venvs/frost-mhd/bin/python experiments/e10_cfd/batch_sweep.py --data data/cfd/mhd256f.h5 \\
      --classical 64:3:none:0:t --learned mhd256_32k8lv_s1:8:h --tag mhd
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np

from frost.cfd import spectral3d as s3, split3d
from frost.cfd.spectral3d import Grid3

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats3d import DT_C, MODELS, classical_advance  # noqa: E402
from w1_eval import RES, best_of  # noqa: E402
from w3split import load_model  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--classical", default="")
    ap.add_argument("--learned", default="")
    ap.add_argument("--batches", default="1,4,16,64")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    with h5py.File(args.data, "r") as fh:
        attrs = dict(fh.attrs)
        ev_ic = np.asarray(fh["eval/u_ic"])
    nu, power = float(attrs["nu"]), float(attrs["power"])
    power_b = float(attrs.get("power_b", -1.0))
    physics = str(attrs.get("physics", "hydro"))
    g_ic = Grid3(int(attrs["n_ic"]), nu=nu, power=power, power_b=power_b, dealias="exp")
    u_ic = s3.to_hat(g_ic, jnp.asarray(ev_ic))
    batches = [int(b) for b in args.batches.split(",")]
    rows = {}

    def tile(u0, b):
        return jnp.tile(u0, (-(-b // u0.shape[0]), 1, 1, 1, 1))[:b]

    for run in (r for r in args.classical.split(",") if r):
        n, m, kind, c, *opt = run.split(":")
        n, m, c = int(n), int(m), float(c)
        g = Grid3(n, nu=nu, power=power, power_b=power_b, dealias="exp", filter_strength=6.0 / m if "t" in opt else 6.0)
        # above the stored initial grid, the stored state zero-padded (timing rows; the 2D frontier does the same)
        u0 = s3.truncate(g_ic, g, u_ic) if n <= g_ic.n else s3.pad(g_ic, g, u_ic)
        rows[f"classical {run}"] = {}
        for b in batches:
            ub = tile(u0, b)
            lo, _ = best_of(lambda ub=ub, g=g: classical_advance(g, ub, kind=kind, c=c, m=m, n_steps=args.steps,
                                                                 physics=physics))
            rows[f"classical {run}"][b] = 1e3 * lo / args.steps / b
    for run in (r for r in args.learned.split(",") if r):
        name, k, *mode = run.split(":")
        model = load_model(MODELS / name)
        if "h" in mode:
            model["static"] = dataclasses.replace(model["static"], net_dtype="bf16")
        meta = json.loads((MODELS / f"{name}.json").read_text())
        g = Grid3(int(meta["grid"]), nu=nu, power=power, power_b=power_b, dealias=meta.get("dealias", "exp"))
        u0 = s3.truncate(g_ic, g, u_ic)
        rows[f"learned {run}"] = {}
        for b in batches:
            ub = tile(u0, b)
            lo, _ = best_of(lambda ub=ub, g=g, model=model: split3d.advance(g, model, ub, k=int(k), dt=DT_C,
                                                                             n_steps=args.steps))
            rows[f"learned {run}"][b] = 1e3 * lo / args.steps / b
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"batch_sweep_{args.tag}.json").write_text(json.dumps(rows, indent=1))
    names = list(rows)
    print("ms per 0.04 per trajectory, by batch:", flush=True)
    for nme in names:
        print(f"  {nme:34s} " + "  ".join(f"b{b}: {rows[nme][b]:7.3f}" for b in batches), flush=True)
    cl = [n for n in names if n.startswith("classical")]
    le = [n for n in names if n.startswith("learned")]
    for a in le:
        for c in cl:
            print(f"  speedup {a} vs {c}: " + "  ".join(f"b{b}: {rows[c][b] / rows[a][b]:5.2f}x" for b in batches))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
