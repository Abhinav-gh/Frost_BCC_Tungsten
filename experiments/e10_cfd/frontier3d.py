"""The classical frontier in 3D (W3, forced isotropic turbulence): filtered spectral at each
resolution and time step, alone and with the classical LES closures, scored and timed.

Trap 22: a learned solver is only compared against the classical method at its own best
settings. Here that means the largest stable step per grid, the exponential filter, and
the standard closures of spectral LES:
- Chollet-Lesieur spectral eddy viscosity (``cl``, c scales nu_t);
- Smagorinsky (``smag``, c = c_s).
tau as in ``w3split.py`` (velocity correlation on the 32^3 band, first dip below 0.95,
eval window 6 time units). Each run starts from the DNS state truncated to its own grid.

Output: data/results/e10/frontier3d_<tag>.json

Usage (guarded launcher; GPU, JAX env)
  scripts/safe_run.sh cfd_frontier3d env -u LD_LIBRARY_PATH -u CUDA_HOME PYTHONPATH=src \\
      .venvs/frost-mhd/bin/python experiments/e10_cfd/frontier3d.py --tag base
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

from frost.cfd import spectral3d as s3
from frost.cfd.spectral3d import Grid3

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w1_eval import RES, best_of, correlation_times  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "cfd" / "hit192.h5"
DT_C = 0.04
DEFAULT = ("32:1:none:0,32:1:cl:0.5,32:1:cl:1,32:1:cl:2,32:1:smag:0.1,32:1:smag:0.17,40:1:none:0,40:1:cl:1,"
           "48:1:none:0,48:1:cl:0.5,48:1:cl:1,48:1:cl:2,48:1:smag:0.1,48:1:smag:0.17,56:1:none:0,56:2:none:0,"
           "64:1:none:0,64:2:none:0,64:1:cl:1,64:2:cl:1,80:2:none:0,96:2:none:0,96:3:none:0")


@partial(jax.jit, static_argnames=("grid", "score", "kind", "c", "m", "n_records"))
def rollout(grid: Grid3, score: Grid3, u0, *, kind: str, c: float, m: int, n_records: int):
    tend = s3.les_tendency(grid, kind, c)

    def record(u, _):
        u = jax.lax.fori_loop(0, m, lambda _, x: s3.step(grid, x, DT_C / m, tend), u)
        return u, (u if grid == score else s3.truncate(grid, score, u))

    return jax.lax.scan(record, u0, None, length=n_records)[1]


@partial(jax.jit, static_argnames=("grid", "kind", "c", "m", "n_steps"))
def advance(grid: Grid3, u0, *, kind: str, c: float, m: int, n_steps: int):
    tend = s3.les_tendency(grid, kind, c)
    return jax.lax.fori_loop(0, n_steps * m, lambda _, x: s3.step(grid, x, DT_C / m, tend), u0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default=DEFAULT, help="n:substeps per 0.04:closure:c,...")
    ap.add_argument("--batches", default="16,64")
    ap.add_argument("--steps", type=int, default=32, help="timed length, in steps of 0.04")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    with h5py.File(args.data, "r") as fh:
        attrs = dict(fh.attrs)
        ev = np.asarray(fh["eval/u32"])
        ev_ic = np.asarray(fh["eval/u_ic"])
    nu, power = float(attrs["nu"]), float(attrs["power"])
    g32 = Grid3(32, nu=nu, power=power, dealias="exp")
    g_ic = Grid3(int(attrs["n_ic"]), nu=nu, power=power, dealias="exp")
    u_ic = s3.to_hat(g_ic, jnp.asarray(ev_ic))
    del ev_ic
    n_rec = ev.shape[0] - 1
    truth = jnp.asarray(ev[1:])
    batches = [int(b) for b in args.batches.split(",") if b]
    print(f"devices {jax.devices()} | eval {ev.shape} | ic {g_ic.n}^3 | batches {batches}", flush=True)

    rows = {}
    for run in args.runs.split(","):
        n, m, kind, c = run.split(":")
        n, m, c = int(n), int(m), float(c)
        g = Grid3(n, nu=nu, power=power, dealias="exp")
        u0 = s3.truncate(g_ic, g, u_ic)
        st = rollout(g, g32, u0, kind=kind, c=c, m=m, n_records=n_rec)
        rho = np.asarray(s3.correlation(s3.to_real(g32, st), truth))
        tau = correlation_times(np.where(np.isfinite(rho), rho, -1.0), DT_C)
        row = {"n": n, "m": m, "dt": DT_C / m, "closure": kind, "c": c, "tau": tau.tolist(),
               "tau_mean": float(tau.mean()), "tau_sem": float(tau.std(ddof=1) / np.sqrt(tau.size)),
               "finite": bool(np.isfinite(rho).all()), "ms_per_0.04": {}}
        del st
        for b in batches:
            ub = jnp.tile(u0, (-(-b // u0.shape[0]), 1, 1, 1, 1))[:b]
            lo, hi = best_of(lambda ub=ub, g=g, kind=kind, c=c, m=m: advance(g, ub, kind=kind, c=c, m=m,
                                                                             n_steps=args.steps))
            row["ms_per_0.04"][str(b)] = 1e3 * lo / args.steps
            row.setdefault("window_ratio", {})[str(b)] = lo / hi
        rows[run] = row
        print(f"{run:18s} tau {row['tau_mean']:.3f} +- {row['tau_sem']:.3f} | finite {row['finite']} | "
              + " | ".join(f"batch {b}: {row['ms_per_0.04'][str(b)]:.2f} ms" for b in batches) + " per 0.04",
              flush=True)

    RES.mkdir(parents=True, exist_ok=True)
    out = RES / f"frontier3d_{args.tag}.json"
    out.write_text(json.dumps({"rows": rows, "steps": args.steps}, indent=1))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
