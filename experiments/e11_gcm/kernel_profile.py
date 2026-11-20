"""G1, step 0b: where the GPU time of one NeuralGCM model step goes, kernel by kernel.

``profile_split.py`` found the dynamical core at ~74% of a 2.8-degree step and running at
a few percent of the A6000's peak. This traces the jitted step with JAX's profiler (CUPTI),
reads the Perfetto trace, and sums device-kernel time by class: GEMMs (cuBLAS/cuBLASLt,
CUTLASS), FFTs, XLA fusions, copies/transposes and the rest; and by the jit's named scopes
(``inv_legendre``, ``fwd_fourier``... from dinosaur's ``named_call``) where XLA keeps them.

Output: data/results/e11/kernels_<tag>.json

Usage (guarded launcher; GPU, frost-gcm env)
  scripts/safe_run.sh gcm_kernels env -u LD_LIBRARY_PATH -u CUDA_HOME PYTHONPATH=src \\
      .venvs/frost-gcm/bin/python experiments/e11_gcm/kernel_profile.py --model data/gcm/models/deterministic_2_8_deg.pkl --tag 2p8
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import pickle
import re
import tempfile
from pathlib import Path

import jax
import neuralgcm

from frost.gcm.neuralgcm_split import SplitModel, fft_transforms

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "data" / "results" / "e11"


def kernel_class(name: str) -> str:
    n = name.lower()
    if any(s in n for s in ("gemm", "cublas", "cutlass", "sm80_xmma", "sm86_xmma", "ampere_", "matmul", "dot")):
        return "gemm"
    if "fft" in n:
        return "fft"
    if "transpose" in n or "copy" in n or "memcpy" in n or "memset" in n:
        return "copy/transpose"
    if "fusion" in n or n.startswith(("loop_", "input_", "wrapped_")):
        return "fusion"
    if "reduce" in n:
        return "reduce"
    return "other"


SCOPES = ("inv_legendre", "fwd_legendre", "inv_fourier", "fwd_fourier", "implicit_inverse", "implicit_terms",
          "explicit_terms", "filter", "physics", "corrector", "encode", "decode")


def scope_label(op_name: str) -> str:
    """The innermost known scope in an HLO op_name path (dinosaur's named_calls, Haiku modules)."""
    parts = op_name.split("/")
    for part in reversed(parts):
        for s in SCOPES:
            if s in part:
                return s
    return parts[-1][:40] if op_name else "(no metadata)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--part", choices=("fused", "physics", "dycore"), default="fused")
    ap.add_argument("--fft", action="store_true", help="our FFT longitude transform (frost.gcm.transforms)")
    ap.add_argument("--fourier", choices=("fft", "matmul"), default=None,
                    help="fourier_method for FastSphericalHarmonics checkpoints (the 0.7 degree model)")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    with open(args.model, "rb") as fh:
        base = f"RealSphericalHarmonicsWithZeroImag.fourier_method = '{args.fourier}'" if args.fourier else ""
        m = SplitModel(pickle.load(fh), "\n".join(o for o in (fft_transforms() if args.fft else "", base) if o))
    params = m.params
    data = neuralgcm.demo.load_data(m.api.data_coords)
    forcings = m.api.forcings_from_xarray(data.isel(time=0))
    x0 = m.api.encode(m.api.inputs_from_xarray(data.isel(time=0)), forcings, jax.random.key(0))
    f0 = m.forcing_at(params, forcings, x0)
    t0 = m.physics(params, x0, f0)

    def fused(x, _):
        return m.advance(params, x, m.forcing_at(params, forcings, x)), None

    def dycore(x, _):
        return m.finish(params, x, t0, f0), None

    if args.part == "physics":
        run = jax.jit(lambda x: jax.lax.map(lambda _: m.physics(params, x, f0), jax.numpy.arange(args.steps)))
    else:
        body = fused if args.part == "fused" else dycore
        run = jax.jit(lambda x: jax.lax.scan(body, x, None, length=args.steps)[0])
    # the optimized HLO maps each fusion and GEMM name to the op_name scopes it came from
    hlo = run.lower(x0).compile().as_text()
    scope_of = {}
    for line in hlo.splitlines():
        head = re.match(r"\s*(?:ROOT )?%?([\w.\-]+) = ", line)
        meta = re.search(r'op_name="([^"]+)"', line)
        if head and meta:
            scope_of.setdefault(head.group(1), meta.group(1))
            scope_of.setdefault(head.group(1).replace(".", "_"), meta.group(1))  # kernel names use "_"
    jax.block_until_ready(run(x0))
    with tempfile.TemporaryDirectory() as tmp:
        with jax.profiler.trace(tmp, create_perfetto_trace=True):
            jax.block_until_ready(run(x0))
        path = next(Path(tmp).rglob("*.trace.json.gz"))
        trace = json.loads(gzip.decompress(path.read_bytes()))

    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    # device-kernel events live on the GPU process; find its pid from the process names
    gpu_pids = {e["pid"] for e in events if e.get("ph") == "M" and e.get("name") == "process_name"
                and "/device:GPU" in str(e.get("args", {}).get("name", ""))}
    by_class, by_name, by_scope = collections.Counter(), collections.Counter(), collections.Counter()
    total = 0.0
    for e in events:
        if e.get("ph") != "X" or e.get("pid") not in gpu_pids:
            continue
        dur = float(e.get("dur", 0.0))  # microseconds
        name = e.get("name", "")
        by_class[kernel_class(name)] += dur
        by_name[name] += dur
        by_scope[scope_label(scope_of.get(name, ""))] += dur
        total += dur
    per_step = total / args.steps / 1e3
    print(f"{args.part}: {per_step:.2f} ms of device kernels per step "
          f"({len(by_name)} distinct kernels)", flush=True)
    for k, v in by_class.most_common():
        print(f"  {k:16s} {100 * v / total:5.1f}%  {v / args.steps / 1e3:.2f} ms", flush=True)
    print("  by source scope:", flush=True)
    for k, v in by_scope.most_common(14):
        print(f"    {100 * v / total:5.1f}%  {v / args.steps / 1e3:.2f} ms  {k}", flush=True)
    print("  top kernels:", flush=True)
    for k, v in by_name.most_common(15):
        print(f"    {100 * v / total:5.1f}%  {k[:110]}", flush=True)

    RES.mkdir(parents=True, exist_ok=True)
    out = RES / f"kernels_{args.tag}_{args.part}.json"
    out.write_text(json.dumps({"ms_per_step": per_step, "by_class_us": dict(by_class),
                               "by_scope_us": dict(by_scope), "top": by_name.most_common(40)}, indent=1))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
