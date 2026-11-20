"""N5: certify a faster CFD / MHD hold without the DNS, by canary members inside the ensemble.

Trusted group: the schedule validated against the DNS (``stats3d.py``); candidate group: a faster or
different schedule. Both start from the same DNS states truncated to the grid, run ``--horizon``
time units through ``frost.runtime`` (``HoldBackend`` + ``Executor``), and are compared on each
member's time-averaged statistics after ``--burn-in`` (``runtime.canary.Canary``). The verdict is
then checked against the DNS-based one already measured for the same pair (stats3d results), which
is what this experiment validates: does the run-time canary refuse the schedules the DNS refuses
and admit the ones it admits?

Statistics per member (on the 32^3 band, shells 1-12): band energy and band enstrophy per field, the
top shells' energy (9-12), the velocity-derivative skewness and flatness on the band, and for MHD
the field's band energy, current and derivative flatness.

Output: data/results/e13/canary_<tag>.json

Usage (guarded launcher; GPU, JAX env)
  scripts/safe_run.sh canary env -u LD_LIBRARY_PATH -u CUDA_HOME PYTHONPATH=src \\
      .venvs/frost-mhd/bin/python experiments/e13_runtime/canary_cfd.py --data data/cfd/mhd256f.h5 \\
      --trusted mhd256_32k4lv:4 --candidates mhd256_32k8lv:8,mhd256_32k4:4 --tag mhd
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from frost.cfd import spectral3d as s3
from frost.cfd.spectral3d import Grid3
from frost.runtime import Executor, NonFinite
from frost.runtime.canary import Canary
from frost.runtime.hold import HoldBackend

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "e10_cfd"))
from stats3d import BLOCK, GRAD_BAND, KB, band_stats  # noqa: E402
from w3split import load_model  # noqa: E402

MODELS = ROOT / "data" / "models" / "cfd"
OUT = ROOT / "data" / "results" / "e13"
DT_C = 0.04
TOL = 0.05  # the largest admissible relative difference of a group mean (5%)


def derived(raw: jnp.ndarray, mhd: bool):
    """Per member: the canary's statistics from ``band_stats`` (shells 1..KB, then moments)."""
    k = jnp.arange(1, KB + 1, dtype=jnp.float32)
    keep = (k <= GRAD_BAND).astype(jnp.float32)

    def one(block):
        e = block[..., :KB]
        m2, m3, m4 = block[..., KB + 3], block[..., KB + 4], block[..., KB + 5]  # the |k| <= 12 moments
        return [jnp.sum(e * keep, -1), jnp.sum(e * keep * k**2, -1), jnp.sum(e[..., 8:12], -1),
                m3 / m2**1.5, m4 / m2**2]

    out = one(raw[..., :BLOCK])
    if mhd:
        b = one(raw[..., BLOCK:2 * BLOCK])
        out += [b[0], b[1], b[4]]
    return jnp.stack(out, -1)


NAMES_U = ["energy", "enstrophy", "top_shells", "skewness", "flatness"]
NAMES_B = ["mag_energy", "current", "mag_flatness"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--trusted", required=True, help="model:k")
    ap.add_argument("--candidates", required=True, help="model:k,...")
    ap.add_argument("--horizon", type=float, default=40.0)
    ap.add_argument("--burn-in", type=float, default=10.0)
    ap.add_argument("--every", type=int, default=10, help="steps of 0.04 between records")
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--starts", default="eval",
                    help="eval: the file's eval/u_ic; train:N: N states per file spread over train/u32's trajectories and times "
                         "(decorrelated starts for a wider ensemble; the canary compares schedules, not the DNS)")
    ap.add_argument("--extra-data", default="", help="comma list of more DNS files whose starts join the ensemble")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    with h5py.File(args.data, "r") as fh:
        attrs = dict(fh.attrs)
    starts = []
    for path in [str(args.data)] + [f for f in args.extra_data.split(",") if f]:
        with h5py.File(path, "r") as fh:
            if args.starts == "eval":
                starts.append(("ic", np.asarray(fh["eval/u_ic"])))
            else:
                n_per = int(args.starts.split(":")[1])
                tr = fh["train/u32"]  # (records, trajectories, components, 32, 32, 32)
                n_rec, n_tr = tr.shape[0], tr.shape[1]
                n_t = max(1, -(-n_per // n_tr))
                times = np.linspace(0, n_rec - 1, n_t + 2)[1:-1].round().astype(int)
                picks = [(t, j) for t in times for j in range(n_tr)][:n_per]
                starts.append(("u32", np.stack([np.asarray(tr[t, j]) for t, j in picks])))
    nu, power = float(attrs["nu"]), float(attrs["power"])
    power_b = float(attrs.get("power_b", -1.0))
    mhd = str(attrs.get("physics", "hydro")) == "mhd"
    g_ic = Grid3(int(attrs["n_ic"]), nu=nu, power=power, power_b=power_b, dealias="exp")
    g32 = Grid3(32, nu=nu, power=power, power_b=power_b, dealias="exp")
    if args.starts != "eval":
        g_ic = g32  # train snapshots are stored on the 32^3 band: start there (models here run on 32^3 or are padded up)
    u_ic = jnp.concatenate([s3.to_hat(g_ic, jnp.asarray(x)) for _, x in starts], axis=0)
    print(f"ensemble: {u_ic.shape[0]} members ({args.starts}, {1 + len([f for f in args.extra_data.split(',') if f])} files)",
          flush=True)
    names = NAMES_U + (NAMES_B if mhd else [])
    stats_fn = jax.jit(lambda u: derived(band_stats(g32, s3.truncate(g32, g32, u)), mhd))
    steps = round(args.horizon / DT_C)
    burn = round(args.burn_in / DT_C)

    def group(spec):
        name, k = spec.split(":")
        model = load_model(MODELS / name)
        meta = json.loads((MODELS / f"{name}.json").read_text())
        g = Grid3(int(meta["grid"]), nu=nu, power=power, power_b=power_b, dealias=meta.get("dealias", "exp"))
        backend = HoldBackend(g, model, s3.truncate(g_ic, g, u_ic), dt=DT_C)
        return backend, Executor(backend, [NonFinite()], k=int(k))

    report = {"data": str(args.data), "trusted": args.trusted, "tolerance": args.tol, "members": int(u_ic.shape[0]),
              "horizon": args.horizon, "burn_in": args.burn_in, "candidates": {}}
    for cand in args.candidates.split(","):
        canary = Canary(group(args.trusted), group(cand), stats_fn, names, burn_in=burn,
                        tolerance={n: args.tol for n in names})
        timeline = []
        import time as _time
        t_start = _time.perf_counter()
        for t in range(args.every, steps + 1, args.every):
            canary.advance(t, args.every)
            if t % (10 * args.every) == 0 and t > burn:
                v = canary.verdict()
                timeline.append({"t": t * DT_C, "verdict": v["verdict"], "refused_on": v.get("refused_on", [])})
        final = canary.verdict()
        wall = _time.perf_counter() - t_start
        report["candidates"][cand] = {"final": final, "timeline": timeline, "wall_s_both_groups": wall}
        print(f"{cand:24s} vs {args.trusted}: {final['verdict']}"
              + (f" on {final['refused_on']}" if final.get("refused_on") else "")
              + " | " + " ".join(f"{n} {s['rel_diff']:+.3f}+-{s['rel_se']:.3f}" for n, s in final["stats"].items()),
              flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"canary_{args.tag}.json").write_text(json.dumps(report, indent=1))
    print(f"-> {OUT / f'canary_{args.tag}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
