"""How much of a coarse solver's correlation time is lost to its initial condition, and
how much to its dynamics (W1, Kolmogorov flow at Re 1000).

Every method in the W1 comparison starts from the reference state truncated to its own
grid. Run a near-converged solver (filtered spectral at 256^2 or 512^2) from the state
truncated to n_ic: its tau is the most any n_ic-started model can reach without
inferring the missing scales, and

    tau(n_ic grid solver)  <  tau(converged dynamics from the n_ic state)  <=  ...

splits the n_ic solver's deficit into a dynamics part (what a learned correction can
recover) and an initial-condition part (what it cannot, short of inferring subgrid
state from the resolved field).

Scored as everywhere in W1: vorticity correlation on the 64^2 2/3 band against the
1024^2 DNS, time to the first dip below 0.95.

Output: data/results/e10/ceiling.json

Usage (guarded launcher; GPU, JAX env)
  scripts/safe_run.sh cfd_ceiling env -u LD_LIBRARY_PATH -u CUDA_HOME PYTHONPATH=src \\
      .venvs/frost-mhd/bin/python experiments/e10_cfd/ceiling.py
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

from frost.cfd import schedule, spectral
from frost.cfd.spectral import Grid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_closure import DATA  # noqa: E402
from w1_eval import RES, correlation_times  # noqa: E402

DT, REC = 0.007, 16


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="64:256:0.0035,64:256:0.00175,64:512:0.00175,96:256:0.0035,"
                    "128:256:0.0035,128:512:0.00175", help="n_ic:n_dyn:dt,...")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    g64, g128 = Grid(64), Grid(128)
    with h5py.File(args.data, "r") as fh:
        ev = np.asarray(fh["eval/w64"])
        ev128 = np.asarray(fh["eval/w128"])
    n_rec = ev.shape[0] - 1
    truth = jnp.asarray(ev[1:])
    w128 = jnp.asarray(np.fft.rfft2(ev128[0]) * g128.mask, jnp.complex64)
    print(f"devices {jax.devices()} | eval {ev.shape} | {n_rec} records of {REC * DT:.3f}", flush=True)

    rows = {}
    for run in args.runs.split(","):
        n_ic, n_dyn, dt = run.split(":")
        n_ic, n_dyn, dt = int(n_ic), int(n_dyn), float(dt)
        g_ic = g128 if n_ic == 128 else Grid(n_ic, dealias="exp")
        g_dyn = Grid(n_dyn, dealias="exp")
        w_ic = w128 if n_ic == 128 else spectral.truncate(g128, g_ic, w128)
        w0 = spectral.pad(g_ic, g_dyn, w_ic)
        per_record = round(REC * DT / dt)
        states = schedule.rollout(g_dyn, None, w0, mode="none", k=1, dt=dt, n_records=n_rec,
                                  record_every=per_record)
        rho = np.asarray(spectral.correlation(spectral.to_real(g64, spectral.truncate(g_dyn, g64, states)), truth))
        tau = correlation_times(np.where(np.isfinite(rho), rho, -1.0), REC * DT)
        se = float(tau.std(ddof=1) / np.sqrt(tau.size))
        rows[run] = {"n_ic": n_ic, "n_dyn": n_dyn, "dt": dt, "tau": float(tau.mean()), "tau_se": se,
                     "tau_each": tau.tolist(), "finite": bool(np.isfinite(rho).all())}
        print(f"ic {n_ic:4d} -> dynamics {n_dyn:4d} at dt {dt:.5f}: tau {tau.mean():.2f} +- {se:.2f} | "
              f"finite {rows[run]['finite']}", flush=True)

    out = RES / f"ceiling{'_' + args.tag if args.tag else ''}.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
