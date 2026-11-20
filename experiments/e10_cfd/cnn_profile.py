"""Where does a W1-FV step go, and how fast can its CNN run? (e2e profile)

At batch 16 the CNN is 93% of a B0 step (1.61 ms). This times the pieces and the
CNN at each available precision, on the same 6 x 64 network:
- float32 at Precision.HIGHEST (true fp32);
- float32 at DEFAULT (TF32 tensor cores on Ampere where XLA chooses them);
- bfloat16 weights and activations, float32 accumulation.

For each it reports ms per call and the achieved TFLOP/s against the network's
FLOP count. Output: data/results/e10/cnn_profile.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from frost.cfd import closure
from frost.cfd import finite_volume as fv

ROOT = Path(__file__).resolve().parents[2]


def bench(fn, *args, reps: int = 50) -> float:
    fn(*args).block_until_ready()
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(reps):
            out = fn(*args)
        out.block_until_ready()
        best = min(best, (time.perf_counter() - t0) / reps)
    return 1e3 * best


def cnn(params, x, precision, dtype):
    pad = 1
    for i, p in enumerate(params):
        x = jnp.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="wrap").astype(dtype)
        x = jax.lax.conv_general_dilated(x, p["w"].astype(dtype), (1, 1), "VALID",
                                         dimension_numbers=("NHWC", "HWIO", "NHWC"),
                                         precision=precision,
                                         preferred_element_type=jnp.float32) + p["b"]
        if i < len(params) - 1:
            x = jax.nn.gelu(x)
    return x


def cnn_gemm(params, x, dtype):
    """Each 3x3 conv as one im2col GEMM: (B*H*W, 9 C_in) x (9 C_in, C_out), on tensor cores."""
    for i, p in enumerate(params):
        b, h, w, c = x.shape
        xp = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="wrap")
        cols = jnp.concatenate([xp[:, a:a + h, d:d + w, :] for a in range(3) for d in range(3)], axis=-1)
        wmat = p["w"].reshape(9 * c, -1)
        x = jax.lax.dot_general(cols.astype(dtype).reshape(-1, 9 * c), wmat.astype(dtype),
                                (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)
        x = x.reshape(b, h, w, -1) + p["b"]
        if i < len(params) - 1:
            x = jax.nn.gelu(x)
    return x


def main() -> int:
    g = fv.FVGrid(64)
    params = closure.init_cnn(jax.random.PRNGKey(0), hidden=64, layers=6, out_ch=fv.N_LI)
    params[-1]["w"] = jax.random.normal(jax.random.PRNGKey(3), params[-1]["w"].shape) * 0.05  # non-trivial
    flops = closure.cnn_flops(params, 64)
    rows = {}
    for b in (1, 16, 64):
        x = jax.random.normal(jax.random.PRNGKey(1), (b, 64, 64, 3))
        w = jax.random.normal(jax.random.PRNGKey(2), (b, 64, 64))
        r = {}
        for name, prec, dt in (("fp32_highest", jax.lax.Precision.HIGHEST, jnp.float32),
                               ("fp32_default", jax.lax.Precision.DEFAULT, jnp.float32),
                               ("bf16", jax.lax.Precision.DEFAULT, jnp.bfloat16)):
            f = jax.jit(lambda p, x, prec=prec, dt=dt: cnn(p, x, prec, dt))
            ms = bench(f, params, x)
            r[name] = {"ms": ms, "tflops": flops * b / (ms * 1e-3) / 1e12}
        with jax.default_matmul_precision("tensorfloat32"):
            f = jax.jit(lambda p, x: cnn(p, x, None, jnp.float32))
            ms = bench(f, params, x)
            r["conv_tf32"] = {"ms": ms, "tflops": flops * b / (ms * 1e-3) / 1e12}
            f = jax.jit(lambda p, x: cnn_gemm(p, x, jnp.float32))
            ms = bench(f, params, x)
            r["gemm_tf32"] = {"ms": ms, "tflops": flops * b / (ms * 1e-3) / 1e12}
        f = jax.jit(lambda p, x: cnn_gemm(p, x, jnp.bfloat16))
        ms = bench(f, params, x)
        r["gemm_bf16"] = {"ms": ms, "tflops": flops * b / (ms * 1e-3) / 1e12}
        ref = jax.jit(lambda p, x: cnn(p, x, jax.lax.Precision.HIGHEST, jnp.float32))(params, x)
        for name, fn in (("gemm_fp32_check", lambda p, x: cnn_gemm(p, x, jnp.float32)),):
            with jax.default_matmul_precision("highest"):
                got = jax.jit(fn)(params, x)
            r[name] = {"ms": 0.0, "tflops": float(jnp.max(jnp.abs(got - ref)) / jnp.max(jnp.abs(ref)))}
        held = fv.li_coefficients(g, {"params": params, "scale": {"u": 1.0, "w": 5.0, "bound": 0.25}}, w)
        step = jax.jit(lambda w, c: fv.step(g, w, 0.007, "li", c, "upwind3"))
        r["solver_step_ms"] = bench(step, w, held)
        feats = jax.jit(lambda w: fv.features(g, w, {"u": 1.0, "w": 5.0}))
        r["features_ms"] = bench(feats, w)
        rows[str(b)] = r
        print(f"batch {b}: " + " | ".join(
            f"{k} {v['ms']:.3f} ms ({v['tflops']:.1f} TFLOP/s)" if isinstance(v, dict) else f"{k} {v:.3f} ms"
            for k, v in r.items()), flush=True)
    out = ROOT / "data" / "results" / "e10" / "cnn_profile.json"
    out.write_text(json.dumps({"gflop_per_sample": flops / 1e9, "rows": rows}, indent=1))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
