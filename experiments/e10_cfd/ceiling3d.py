"""The initial-condition ceiling in 3D (W3, forced isotropic turbulence): ``ceiling.py``'s
measurement on ``frost.cfd.spectral3d``.

Run converged dynamics (filtered spectral at 128^3, the band of the 192^3 DNS) from the
eval state truncated to n_ic; its tau is the most any n_ic-started model can reach without
inferring the missing scales. Intermediate dynamics resolutions show how much of the gap
a finer grid's dynamics closes from the same start. Scored as in ``w3split.py``: velocity
correlation on the 32^3 band against the DNS, time to the first dip below 0.95.

Output: data/results/e10/ceiling3d.json

Usage (guarded launcher; GPU, JAX env)
  scripts/safe_run.sh cfd_ceiling3d env -u LD_LIBRARY_PATH -u CUDA_HOME PYTHONPATH=src \\
      .venvs/frost-mhd/bin/python experiments/e10_cfd/ceiling3d.py
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
from w1_eval import RES, correlation_times  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "cfd" / "hit192.h5"
DT_C = 0.04


@partial(jax.jit, static_argnames=("grid", "score", "dt", "n_records", "record_every"))
def rollout32(grid: Grid3, score: Grid3, u0, *, dt: float, n_records: int, record_every: int):
    """Classical rollout keeping only each record's 32^3 band (a full-grid history does not fit)."""

    def record(u, _):
        u = jax.lax.fori_loop(0, record_every, lambda _, x: s3.step(grid, x, dt), u)
        return u, s3.truncate(grid, score, u)

    return jax.lax.scan(record, u0, None, length=n_records)[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="32:32:1,32:64:3,32:128:4,48:48:1,48:64:2,48:128:4,64:64:2,64:96:3,64:128:4",
                    help="n_ic:n_dyn:substeps per 0.04,...")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--tag", default="")
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
    print(f"devices {jax.devices()} | eval {ev.shape} | ic {g_ic.n}^3 | {n_rec} records of {DT_C}", flush=True)

    rows = {}
    for run in args.runs.split(","):
        n_ic, n_dyn, m = (int(x) for x in run.split(":"))
        gi = Grid3(n_ic, nu=nu, power=power, dealias="exp")
        gd = Grid3(n_dyn, nu=nu, power=power, dealias="exp")
        u0 = s3.truncate(g_ic, gi, u_ic)
        u0 = u0 if n_dyn == n_ic else s3.pad(gi, gd, u0)
        st = rollout32(gd, g32, u0, dt=DT_C / m, n_records=n_rec, record_every=m)
        rho = np.asarray(s3.correlation(s3.to_real(g32, st), truth))
        tau = correlation_times(np.where(np.isfinite(rho), rho, -1.0), DT_C)
        se = float(tau.std(ddof=1) / np.sqrt(tau.size))
        rows[run] = {"n_ic": n_ic, "n_dyn": n_dyn, "dt": DT_C / m, "tau": float(tau.mean()), "tau_se": se,
                     "tau_each": tau.tolist(), "finite": bool(np.isfinite(rho).all())}
        print(f"ic {n_ic:4d} -> dynamics {n_dyn:4d} at dt {DT_C / m:.4f}: tau {tau.mean():.3f} +- {se:.3f} | "
              f"finite {rows[run]['finite']}", flush=True)

    out = RES / f"ceiling3d{'_' + args.tag if args.tag else ''}.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
