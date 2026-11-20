"""What an RTX A6000 (GA102, sm_86) actually delivers, per precision and per shape.

Spec sheets quote peaks; Frost's planner needs achieved rates for the shapes it runs. This
measures, in JAX/XLA on one GPU:
- **GEMM throughput** for each float32 dot algorithm XLA:GPU offers on Ampere (true F32 on
  CUDA cores; TF32 tensor cores in 1 and 3 passes; BF16 in 1, 3 and 6 passes, the last
  being dinosaur's default for spherical-harmonic transforms on GPU), at a large square
  shape and at NeuralGCM's transform shapes (batched Legendre: m x [lat, l] @ [l, fields];
  longitude DFT: [lon, lon] @ [lon, fields]);
- **memory bandwidth** (a large elementwise update) against the 768 GB/s of GDDR6;
- **FFT against matmul-DFT** for the 128-point longitude transform (TPUs have no fast FFT,
  so dinosaur uses a DFT matrix; GPUs have cuFFT);
- the **accuracy** of each algorithm against float64 on the transform shape.

Output: data/results/e11/ampere_probe.json

Usage (guarded launcher; GPU, frost-gcm env)
  scripts/safe_run.sh ampere_probe env -u LD_LIBRARY_PATH -u CUDA_HOME PYTHONPATH=src \\
      .venvs/frost-gcm/bin/python experiments/e11_gcm/ampere_probe.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "data" / "results" / "e11"
P = lax.DotAlgorithmPreset
ALGOS = {"f32 (CUDA cores)": P.F32_F32_F32, "tf32 x1": P.TF32_TF32_F32, "tf32 x3": P.TF32_TF32_F32_X3,
         "bf16 x1": P.BF16_BF16_F32, "bf16 x3": P.BF16_BF16_F32_X3, "bf16 x6": P.BF16_BF16_F32_X6}


def timeit(fn, *args, reps: int = 20):
    jax.block_until_ready(fn(*args))
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(reps):
            out = fn(*args)
        jax.block_until_ready(out)
        best = min(best, (time.perf_counter() - t0) / reps)
    return best


def main() -> int:
    rows = {"device": str(jax.devices()[0]), "gemm": {}, "accuracy": {}}
    key = jax.random.key(0)
    shapes = {
        "square 4096": ("ij,jk->ik", (4096, 4096), (4096, 4096)),
        "legendre TL63 (m 64, lat 64, l 65, 640 fields)": ("mjl,mlf->mjf", (64, 64, 65), (64, 65, 640)),
        "legendre TL127 (m 128, lat 128, l 129, 640 fields)": ("mjl,mlf->mjf", (128, 128, 129), (128, 129, 640)),
        "lon DFT 128 (x 64 lat x 640 fields)": ("im,mf->if", (128, 128), (128, 64 * 640)),
    }
    for sname, (sub, sa, sb) in shapes.items():
        ka, kb = jax.random.split(jax.random.fold_in(key, len(rows["gemm"])))
        a = jax.random.normal(ka, sa, jnp.float32)
        b = jax.random.normal(kb, sb, jnp.float32)
        flops = 2.0 * np.prod(jax.eval_shape(lambda a, b: jnp.einsum(sub, a, b), a, b).shape) * sa[-1]
        ref = np.einsum(sub, np.asarray(a, np.float64), np.asarray(b, np.float64)) if "square" not in sname else None
        rows["gemm"][sname] = {}
        for name, algo in ALGOS.items():
            fn = jax.jit(lambda a, b, algo=algo: jnp.einsum(sub, a, b, precision=algo))
            t = timeit(fn, a, b)
            rows["gemm"][sname][name] = {"ms": 1e3 * t, "tflops": flops / t / 1e12}
            line = f"{sname:52s} {name:18s} {1e3 * t:8.3f} ms {flops / t / 1e12:7.2f} TFLOP/s"
            if ref is not None:
                err = float(np.max(np.abs(np.asarray(fn(a, b), np.float64) - ref)) / np.max(np.abs(ref)))
                rows["accuracy"].setdefault(sname, {})[name] = err
                line += f" | max rel err {err:.1e}"
            print(line, flush=True)

    x = jnp.ones((256 * 1024 * 1024,), jnp.float32)  # 1 GiB
    t = timeit(jax.jit(lambda x: x * 1.0001 + 1.0), x)
    rows["bandwidth_GBps"] = 2 * x.nbytes / t / 1e9
    print(f"elementwise read+write over 1 GiB: {rows['bandwidth_GBps']:.0f} GB/s (GDDR6 peak 768)", flush=True)

    f = jax.random.normal(key, (640, 64, 128), jnp.float32)  # fields x lat x lon
    dft = jnp.asarray(np.fft.rfft(np.eye(128), axis=0).real.T, jnp.float32)  # a real 128x65 block, as a stand-in
    t_fft = timeit(jax.jit(lambda f: jnp.fft.rfft(f, axis=-1)), f)
    t_dft = {n: timeit(jax.jit(lambda f, a=a: jnp.einsum("...i,im->...m", f, dft, precision=a)), f)
             for n, a in (("f32", P.F32_F32_F32), ("bf16 x6", P.BF16_BF16_F32_X6))}
    rows["lon_transform_ms"] = {"rfft (cuFFT)": 1e3 * t_fft, **{f"DFT matmul {n}": 1e3 * v for n, v in t_dft.items()}}
    print(f"128-point longitude transform of 640x64 rows: rfft {1e3 * t_fft:.3f} ms | DFT matmul "
          + " | ".join(f"{n} {1e3 * v:.3f} ms" for n, v in t_dft.items()), flush=True)

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "ampere_probe.json").write_text(json.dumps(rows, indent=1))
    print(f"wrote {RES / 'ampere_probe.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
