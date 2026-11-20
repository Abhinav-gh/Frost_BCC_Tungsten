"""G1, step 0: does our split reproduce NeuralGCM, and where does a step's time go on our GPU?

The paper (Methods) says one learned-physics call costs ~1.5x a dynamical-core step on a
TPU, so holding it for 30-60 minutes (5 dycore substeps) left the physics ~23% of the cost.
Frost's levers on the physics only matter if that holds here, so measure it first:

1. **Equivalence.** ``finish(x, physics(x, f), f)`` against the stock fused ``advance``, one
   model step from an encoded ERA5 state (NeuralGCM's bundled TL31 demo data, regridded).
2. **Cost** inside ``lax.scan``, best of three windows: the fused model step; one physics
   call; one held-tendency dycore interval (``finish``: the substeps between physics calls);
   at batch 1 and a vmapped batch, and under XLA's default float32 matmul precision and
   ``highest``.

Output: data/results/e11/profile_<tag>.json

Usage (guarded launcher; GPU, frost-gcm env)
  scripts/safe_run.sh gcm_profile env -u LD_LIBRARY_PATH -u CUDA_HOME PYTHONPATH=src \\
      .venvs/frost-gcm/bin/python experiments/e11_gcm/profile_split.py --model data/gcm/models/deterministic_2_8_deg.pkl --tag 2p8
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import neuralgcm
import numpy as np

from frost.gcm.neuralgcm_split import SplitModel

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "data" / "results" / "e11"


def best_of(fn, windows: int = 3):
    jax.block_until_ready(fn())
    times = []
    for _ in range(windows):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        times.append(time.perf_counter() - t0)
    return min(times), max(times)


def numeric(tree):
    return [v for v in jax.tree_util.tree_leaves(tree)
            if jnp.issubdtype(jnp.asarray(v).dtype, jnp.inexact)]  # PRNG keys and step counters are not compared


def rel_diff(a, b):
    la, lb = numeric(a), numeric(b)
    num = sum(float(jnp.sum((x - y) ** 2)) for x, y in zip(la, lb, strict=True))
    den = sum(float(jnp.sum(y**2)) for y in lb)
    return (num / max(den, 1e-30)) ** 0.5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--batches", default="1,4")
    ap.add_argument("--steps", type=int, default=8, help="model steps per timed window")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    with open(args.model, "rb") as fh:
        ckpt = pickle.load(fh)
    m = SplitModel(ckpt)
    params = m.params
    data = neuralgcm.demo.load_data(m.api.data_coords)
    inputs = m.api.inputs_from_xarray(data.isel(time=0))
    forcings = m.api.forcings_from_xarray(data.isel(time=0))
    x0 = m.api.encode(inputs, forcings, jax.random.key(0))
    print(f"devices {jax.devices()} | model step {m.api.timestep} | physics calls per step {m.num_substeps} | "
          f"grid {m.api.model_coords.horizontal.nodal_shape} x {m.api.model_coords.vertical.layers} levels", flush=True)

    @jax.jit
    def fused(x):
        return m.advance(params, x, m.forcing_at(params, forcings, x))

    @jax.jit
    def split(x):  # one model step: num_substeps physics calls, each followed by its held-tendency dycore
        for _ in range(m.num_substeps):
            f = m.forcing_at(params, forcings, x)
            x = m.finish(params, x, m.physics(params, x, f), f)
        return x

    ref = m.api.advance(x0, forcings)
    eq = {"vs_api_advance": rel_diff(split(x0), ref), "fused_vs_api_advance": rel_diff(fused(x0), ref)}
    print(f"equivalence (relative L2 over the state): split vs api.advance {eq['vs_api_advance']:.2e} | "
          f"our fused vs api.advance {eq['fused_vs_api_advance']:.2e}", flush=True)

    f0 = m.forcing_at(params, forcings, x0)
    tend0 = m.physics(params, x0, f0)
    rows = {}
    for precision in ("default", "highest"):
        for b in (int(v) for v in args.batches.split(",")):
            xb = jax.tree_util.tree_map(lambda a, b=b: jnp.broadcast_to(a, (b, *jnp.shape(a))), x0)
            tb = jax.tree_util.tree_map(lambda a, b=b: jnp.broadcast_to(a, (b, *jnp.shape(a))), tend0)
            with jax.default_matmul_precision(precision):

                @jax.jit
                def run_fused(x):
                    def one(x, _):
                        return jax.vmap(lambda s: m.advance(params, s, m.forcing_at(params, forcings, s)))(x), None
                    return jax.lax.scan(one, x, None, length=args.steps)[0]

                @jax.jit
                def run_physics(x):
                    def one(c, _):
                        t = jax.vmap(lambda s: m.physics(params, s, m.forcing_at(params, forcings, s)))(x)
                        return c + sum(jnp.sum(v) for v in numeric(t)).real, None
                    return jax.lax.scan(one, jnp.zeros(()), None, length=args.steps)[0]

                @jax.jit
                def run_finish(x, t):
                    def one(x, _):
                        return jax.vmap(lambda s, tt: m.finish(params, s, tt, m.forcing_at(params, forcings, s)))(x, t), None
                    return jax.lax.scan(one, x, None, length=args.steps)[0]

                lo_f, hi_f = best_of(lambda: run_fused(xb))
                lo_p, hi_p = best_of(lambda: run_physics(xb))
                lo_d, hi_d = best_of(lambda: run_finish(xb, tb))
            row = {"precision": precision, "batch": b, "ms_step_fused": 1e3 * lo_f / args.steps,
                   "ms_physics_call": 1e3 * lo_p / args.steps,
                   "ms_dycore_held": 1e3 * lo_d / args.steps, "windows": [lo_f / hi_f, lo_p / hi_p, lo_d / hi_d]}
            row["physics_share"] = row["ms_physics_call"] * m.num_substeps / row["ms_step_fused"]
            rows[f"{precision}:b{b}"] = row
            print(f"{precision:8s} batch {b}: fused step {row['ms_step_fused']:.2f} ms | physics call "
                  f"{row['ms_physics_call']:.2f} ms | held dycore {row['ms_dycore_held']:.2f} ms | physics share "
                  f"{100 * row['physics_share']:.0f}% | windows {[round(w, 2) for w in row['windows']]}", flush=True)

    RES.mkdir(parents=True, exist_ok=True)
    out = RES / f"profile_{args.tag}.json"
    out.write_text(json.dumps({"model": str(args.model), "timestep": str(m.api.timestep),
                               "num_substeps": m.num_substeps, "equivalence": eq, "rows": rows}, indent=1))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
