"""End-to-end profile of the learned 3D closure under a hold, and the loop's own overhead.

The failure loop's first step for W3 / W5: where does a held step's time go? At batch 16, per 0.04
time units, on one A6000:
- base: the physics step alone (``split3d.physics_step``, no network);
- net: one correction call (``split3d.correction``), in float32 and with the network in bfloat16;
- held, as shipped: ``split3d.advance`` (one ``lax.cond`` per step decides whether to refresh);
- held, nested: an outer scan over holds whose body calls the network once and then scans the
  k physics steps, with no conditional inside the loop;
- the same with the bfloat16 network.
The model expects ``held = base + net / k``; the gap is loop overhead.

bfloat16 network: the parameters and activations of the encoder-process-decoder are cast to
bfloat16 inside our own forward (``correction_bf16``), the output back to float32 before the
projection. Its relative difference from the float32 correction is reported on the same states.

Output: data/results/e10/hold_profile_<tag>.json

Usage (guarded launcher; GPU, JAX env)
  scripts/safe_run.sh cfd_hold_profile env -u LD_LIBRARY_PATH -u CUDA_HOME PYTHONPATH=src \\
      .venvs/frost-mhd/bin/python experiments/e10_cfd/hold_profile.py --model mhd256_32k4lv --k 4 \\
      --data data/cfd/mhd256f.h5 --tag mhd_k4
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from frost.cfd import spectral3d as s3, split3d
from frost.cfd.spectral3d import Grid3

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w1_eval import RES, best_of  # noqa: E402
from w3split import load_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "data" / "models" / "cfd"
DT_C = 0.04


def _cast(tree, dtype):
    return jax.tree_util.tree_map(lambda a: a.astype(dtype) if hasattr(a, "astype") else a, tree)


def correction_bf16(grid: Grid3, model: dict, u_hat):
    """``split3d.correction`` with the network evaluated in bfloat16 (our own forward)."""
    lead = u_hat.shape[:-4]
    n, ch = grid.n, u_hat.shape[-4]
    u = s3.to_real(grid, u_hat)
    x = (jnp.moveaxis(u, -4, -1).reshape(-1, n, n, n, ch) * model["scale_in"]).astype(jnp.bfloat16)
    params = _cast(model["params"], jnp.bfloat16)
    c = split3d.apply_epd3(params, x, model["static"].rates, model["static"].arch).astype(jnp.float32)
    c = model["scale_out"] * c
    c = jnp.moveaxis(c.reshape(*lead, n, n, n, ch), -1, -4)
    return split3d._project(grid, s3.to_hat(grid, c)).at[..., 0, 0, 0].set(0.0)


def nested_advance(grid: Grid3, model: dict, u_hat, *, k: int, dt: float, n_steps: int, corr):
    """One network call per hold, then k physics steps with it held: no conditional in the loop."""
    base, phys = split3d.base_of(model), split3d.physics_of(model)

    def hold(u, _):
        c = corr(grid, model, u)
        u = jax.lax.fori_loop(0, k, lambda _, x: split3d.split_step(grid, x, c, dt, base, *phys), u)
        return u, None

    return jax.lax.scan(hold, u_hat, None, length=n_steps // k)[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    with h5py.File(args.data, "r") as fh:
        attrs = dict(fh.attrs)
        ev_ic = np.asarray(fh["eval/u_ic"])
    nu, power = float(attrs["nu"]), float(attrs["power"])
    power_b = float(attrs.get("power_b", -1.0))
    model = load_model(MODELS / args.model)
    meta = json.loads((MODELS / f"{args.model}.json").read_text())
    g_ic = Grid3(int(attrs["n_ic"]), nu=nu, power=power, power_b=power_b, dealias="exp")
    g = Grid3(int(meta["grid"]), nu=nu, power=power, power_b=power_b, dealias=meta.get("dealias", "exp"))
    u0 = s3.truncate(g_ic, g, s3.to_hat(g_ic, jnp.asarray(ev_ic)))
    ub = jnp.tile(u0, (-(-args.batch // u0.shape[0]), 1, 1, 1, 1))[:args.batch]
    k, steps = args.k, args.steps
    base, phys = split3d.base_of(model), split3d.physics_of(model)

    phys_only = jax.jit(lambda u: jax.lax.fori_loop(
        0, steps, lambda _, x: split3d.physics_step(g, x, DT_C, base, *phys), u))
    net32 = jax.jit(lambda u: split3d.correction(g, model, u))
    net16 = jax.jit(lambda u: correction_bf16(g, model, u))
    shipped = lambda u: split3d.advance(g, model, u, k=k, dt=DT_C, n_steps=steps)  # noqa: E731
    nest32 = jax.jit(partial(nested_advance, g, model, k=k, dt=DT_C, n_steps=steps, corr=split3d.correction))
    nest16 = jax.jit(partial(nested_advance, g, model, k=k, dt=DT_C, n_steps=steps, corr=correction_bf16))

    res = {}
    for name, fn, per in (("base_ms_per_0.04", phys_only, steps), ("net_fp32_ms_per_call", net32, 1),
                          ("net_bf16_ms_per_call", net16, 1), ("held_shipped_ms_per_0.04", shipped, steps),
                          ("held_nested_fp32_ms_per_0.04", nest32, steps),
                          ("held_nested_bf16_ms_per_0.04", nest16, steps)):
        lo, _ = best_of(lambda fn=fn: fn(ub))
        res[name] = 1e3 * lo / per
    res["model_expect_ms_per_0.04"] = res["base_ms_per_0.04"] + res["net_fp32_ms_per_call"] / k
    c32, c16 = np.asarray(net32(ub)), np.asarray(net16(ub))
    res["bf16_relative_difference"] = float(np.sqrt(np.mean(np.abs(c16 - c32) ** 2) / np.mean(np.abs(c32) ** 2)))
    res.update({"model": args.model, "k": k, "batch": args.batch, "grid": g.n, "physics": phys[0],
                "substeps": phys[1], "devices": [str(d) for d in jax.devices()]})
    RES.mkdir(parents=True, exist_ok=True)
    out = RES / f"hold_profile_{args.tag}.json"
    out.write_text(json.dumps(res, indent=2))
    for key, v in res.items():
        print(f"{key:34s} {v}")
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
