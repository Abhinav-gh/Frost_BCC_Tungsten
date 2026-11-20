"""NeuralGCM through Frost's runtime (``frost.runtime.gcm.GCMBackend`` + ``Executor``).

Parity: the runtime's sync schedule at k = 1 (physics every interval, as released) must reproduce
the released fused ``advance`` over ``--hours`` model hours from an ERA5 start, to round-off. Then it
times the released schedule and the relaxation hold through the same executor, so the fourth domain
runs on the same runtime as MD, CFD and MHD. Skill is certified by ``forecast_skill.py`` (30 starts);
this is the systems path, not the skill test.

Output: data/results/e13/gcm_<tag>.json

Usage (guarded launcher; GPU, frost-gcm env)
  scripts/safe_run.sh gcm_rt env -u LD_LIBRARY_PATH -u CUDA_HOME PYTHONPATH=src .venvs/frost-gcm/bin/python \\
      experiments/e13_runtime/gcm_runtime.py --model data/gcm/models/deterministic_2_8_deg.pkl --tag p28
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import xarray

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "e11_gcm"))
from forecast_skill import parse_variant  # noqa: E402

from frost.gcm.neuralgcm_split import SplitModel  # noqa: E402
from frost.runtime import Executor  # noqa: E402
from frost.runtime.gcm import GCMBackend  # noqa: E402

OUT = ROOT / "data" / "results" / "e13"
DATA = ROOT / "data" / "gcm"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data-tag", default="jf2020")
    ap.add_argument("--hours", type=int, default=48)
    ap.add_argument("--variant", default="sub3+damp+f32", help="configuration for the timed rows")
    ap.add_argument("--fourier", choices=("fft", "matmul"), default=None,
                    help="dinosaur's fourier_method for FastSphericalHarmonics checkpoints (the 0.7 degree model), on "
                         "every row: its GPU default resolves to fft while the config stacks transforms, which "
                         "dinosaur rejects")
    ap.add_argument("--chunks", default="1,6", help="refresh blocks per compiled call to time (0.7 degree: 1; the "
                                                   "6-block scan's compile was killed out of memory)")
    ap.add_argument("--rows", choices=("both", "released", "variant"), default="both",
                    help="which configurations to time; one per process at 0.7 degree (compiling a second one in the "
                         "same process exhausted host RAM)")
    ap.add_argument("--no-parity", action="store_true", help="skip the runtime-against-released parity check")
    ap.add_argument("--variant-gin", default="", help="gin bindings for the variant rows only (a systems change to time "
                                                      "against the released rows, e.g. a transform method)")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    ckpt = pickle.load(open(args.model, "rb"))
    base = f"RealSphericalHarmonicsWithZeroImag.fourier_method = '{args.fourier}'" if args.fourier else ""
    probe = SplitModel(ckpt, base)
    grid = probe.api.data_coords.horizontal
    label = f"{grid.longitude_nodes}x{grid.latitude_nodes}"
    ds0 = xarray.open_dataset(DATA / f"era5_{label}_{args.data_tag}_init.nc").isel(time=0)
    report = {"model": str(args.model), "hours": args.hours, "fourier": args.fourier, "rows": {}}

    def start(m):
        inputs, forcings = m.api.inputs_from_xarray(ds0), m.api.forcings_from_xarray(ds0)
        return m.api.encode(inputs, forcings, jax.random.key(0)), forcings

    # parity: released configuration, runtime sync k = 1 against the fused advance
    m = probe
    if args.no_parity:
        m = None
    if m is not None:
        x0, forcings = start(m)
        adv = jax.jit(lambda x: m.advance(m.params, x, m.forcing_at(m.params, forcings, x)))
        x = x0
        for _ in range(args.hours):  # one released model step is one hour, with num_substeps physics calls inside it
            x = adv(x)
        backend = GCMBackend(m, m.params, x0, forcings)
        Executor(backend, [], k=1).run(args.hours * m.num_substeps)
        a = np.asarray(jax.tree_util.tree_leaves(backend.x.state)[0])
        b = np.asarray(jax.tree_util.tree_leaves(x.state)[0])
        rel = float(np.abs(a - b).max() / np.abs(b).max())
        report["parity_rel_max_diff"] = rel
        print(f"parity: runtime sync k=1 vs released advance over {args.hours} h: rel max diff {rel:.2e}", flush=True)

    chunks = [int(c) for c in args.chunks.split(",")]
    names = {"both": ("released", args.variant), "released": ("released",), "variant": (args.variant,)}[args.rows]
    finals = {}
    for name, k, mode, chunk in [(v, 1, "sync", c) for v in names for c in chunks]:
        overrides, _, _ = parse_variant(name, ckpt["model_config_str"])
        extra = args.variant_gin if name != "released" else ""
        mv = SplitModel(ckpt, "\n".join(o for o in (base, overrides, extra) if o))
        xv, fv = start(mv)
        n_int = args.hours * mv.num_substeps
        be = GCMBackend(mv, mv.params, xv, fv, chunk=chunk)
        ex = Executor(be, [], k=k * chunk, mode=mode)
        ex.run(min(n_int, 2 * k * chunk))  # compile
        be2 = GCMBackend(mv, mv.params, xv, fv, chunk=chunk)
        be2._run = be._run  # reuse the compiled intervals (our own object's cache)
        t0 = time.perf_counter()
        Executor(be2, [], k=k * chunk, mode=mode).run(n_int)
        jax.block_until_ready(be2.x)
        ms = 1e3 * (time.perf_counter() - t0) / args.hours
        row = f"{name}:{mode}{k}:chunk{chunk}"
        report["rows"][row] = {"ms_per_model_hour": ms, "physics_calls": be2.physics_calls,
                               "finite": bool(np.isfinite(be2._mass(be2.x))),
                               "mass_drift": float(be2._mass(be2.x) - be2.mass0)}
        print(f"{row:28s} {ms:7.2f} ms per model hour | physics calls {be2.physics_calls} | mass drift "
              f"{report['rows'][row]['mass_drift']:+.3e}", flush=True)
        finals[name] = np.asarray(jax.tree_util.tree_leaves(be2.x.state)[0])
    if len(finals) == 2:
        # how far the variant's state is from the released one after the same simulated time: ~1e-4 for a change of
        # algorithm with the same numerics (the runtime's own parity), O(1) for a different schedule
        a, b = finals["released"], finals[args.variant]
        report["variant_vs_released_rel_max_diff"] = float(np.abs(b - a).max() / np.abs(a).max())
        print(f"variant against released after {args.hours} h: rel max diff "
              f"{report['variant_vs_released_rel_max_diff']:.2e}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"gcm_{args.tag}.json").write_text(json.dumps(report, indent=1))
    print(f"-> {OUT / f'gcm_{args.tag}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
