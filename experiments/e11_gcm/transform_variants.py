"""G1, step 0c: retarget NeuralGCM's dynamical core to the GPU by configuration alone.

The released checkpoints run every spherical-harmonic transform through dinosaur's
``RealSphericalHarmonics`` (its "pedagogical implementation"), with float32 matmuls resolved
to 6-pass bfloat16 on GPU. ``kernel_profile.py`` found 62% of the dycore's time in fusions
and transposes around those einsums. dinosaur's ``FastSphericalHarmonics`` is a public,
gin-registered alternative with a ``transform_precision`` option; ``frost.gcm.transforms``
adds ours, which keeps the released transform and replaces only its longitude DFT matmul by
an FFT. Per variant:
- **cost**: ms per held-tendency dycore interval and per fused model step (best of three);
- **drift**: the decoded pressure-level forecast after 1 and 24 model steps against the
  released configuration, as relative L2 over every output field.

The decision rule is the certificate's: a variant is admissible if its drift at 24 h is
below the released model's own sensitivity, measured by perturbing the initial state by
float32 round-off (1e-7 relative) and running the released model.

Output: data/results/e11/transform_variants_<tag>.json

Usage (guarded launcher; GPU, frost-gcm env)
  scripts/safe_run.sh gcm_variants env -u LD_LIBRARY_PATH -u CUDA_HOME PYTHONPATH=src \\
      .venvs/frost-gcm/bin/python experiments/e11_gcm/transform_variants.py --model data/gcm/models/deterministic_2_8_deg.pkl --tag 2p8
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

from frost.gcm.neuralgcm_split import SplitModel, f32_transforms, fast_transforms, fft_transforms

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "data" / "results" / "e11"
VARIANTS = {"released (Real, bf16 x6)": "", "fast, platform default": fast_transforms(None),
            "fast, float32": fast_transforms("highest"), "fast, bf16 x3": fast_transforms("tensorfloat32"),
            "fast, bf16 x1": fast_transforms("bfloat16"), "ours: FFT longitude": fft_transforms(),
            "ours: Real in float32": f32_transforms()}


def best_of(fn, windows: int = 3):
    jax.block_until_ready(fn())
    times = []
    for _ in range(windows):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        times.append(time.perf_counter() - t0)
    return min(times)


def rel(a: dict, b: dict) -> float:
    keys = [k for k in b if k in a and jnp.issubdtype(jnp.asarray(b[k]).dtype, jnp.inexact)]
    num = sum(float(jnp.sum((jnp.asarray(a[k]) - jnp.asarray(b[k])) ** 2)) for k in keys)
    den = sum(float(jnp.sum(jnp.asarray(b[k]) ** 2)) for k in keys)
    return (num / den) ** 0.5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--variants", default=";".join(VARIANTS), help="semicolon-separated names; the first is the reference")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    with open(args.model, "rb") as fh:
        ckpt = pickle.load(fh)

    rows, ref = {}, {}
    for name in args.variants.split(";"):
        overrides = VARIANTS[name]
        m = SplitModel(ckpt, overrides)
        params = m.params
        data = neuralgcm.demo.load_data(m.api.data_coords)
        inputs = m.api.inputs_from_xarray(data.isel(time=0))
        forcings = m.api.forcings_from_xarray(data.isel(time=0))

        @jax.jit
        def rollout(x):
            out = []
            for i in range(args.steps):
                x = m.advance(params, x, m.forcing_at(params, forcings, x))
                if i in (0, args.steps - 1):
                    out.append(m.api.decode(x, forcings))
            return out

        @jax.jit
        def dycore(x, t, f):
            return jax.lax.scan(lambda c, _: (m.finish(params, c, t, f), None), x, None, length=8)[0]

        @jax.jit
        def fused(x):
            return jax.lax.scan(lambda c, _: (m.advance(params, c, m.forcing_at(params, forcings, c)), None),
                                x, None, length=8)[0]

        x0 = m.api.encode(inputs, forcings, jax.random.key(0))
        f0 = m.forcing_at(params, forcings, x0)
        t0 = m.physics(params, x0, f0)
        row = {"ms_dycore_interval": 1e3 * best_of(lambda: dycore(x0, t0, f0)) / 8,
               "ms_model_step": 1e3 * best_of(lambda: fused(x0)) / 8}
        out1, out24 = rollout(x0)
        if not ref:
            ref = {"1": out1, str(args.steps): out24}
            # the released model's own sensitivity: the same run from inputs perturbed by round-off
            noisy = jax.tree_util.tree_map(
                lambda v: v * (1.0 + 1e-7 * jax.random.normal(jax.random.key(1), jnp.shape(v)))
                if jnp.issubdtype(jnp.asarray(v).dtype, jnp.floating) else v, inputs)
            n1, n24 = rollout(m.api.encode(noisy, forcings, jax.random.key(0)))
            rows["round-off sensitivity"] = {"drift_1": rel(n1, out1), f"drift_{args.steps}": rel(n24, out24)}
        row["drift_1"], row[f"drift_{args.steps}"] = rel(out1, ref["1"]), rel(out24, ref[str(args.steps)])
        rows[name] = row
        print(f"{name:26s} dycore interval {row['ms_dycore_interval']:6.2f} ms | model step {row['ms_model_step']:6.2f} ms"
              f" | drift 1 step {row['drift_1']:.1e}, {args.steps} steps {row[f'drift_{args.steps}']:.1e}", flush=True)
        if "round-off sensitivity" in rows and len(rows) == 2:
            s = rows["round-off sensitivity"]
            print(f"{'round-off sensitivity':26s} drift 1 step {s['drift_1']:.1e}, {args.steps} steps "
                  f"{s[f'drift_{args.steps}']:.1e}", flush=True)

    RES.mkdir(parents=True, exist_ok=True)
    out = RES / f"transform_variants_{args.tag}.json"
    out.write_text(json.dumps({"model": str(args.model), "rows": rows}, indent=1))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
