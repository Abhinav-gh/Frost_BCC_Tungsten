"""The classical frontier for W1: spectral DS across resolution and time step (the strong baseline).

Frost's learned solver is quoted against spectral DS at 128^2 (tau 6.74 at 0.86 ms per
0.007 time units). That point runs at dt/2, a choice made for CFL headroom, never
measured as the cheapest setting that keeps its accuracy. A weak baseline overstates
any learned solver (McGreivy & Hakim 2024), so this sweeps the strongest classical
family here, our IFRK4 pseudo-spectral solver, over resolution and time step:
- **tau** over the 16 eval trajectories, scored at 64^2 by spectral truncation (as
  ``w1spec.py`` scores DS 128^2), so (64, 0.007) must give 2.48 and (128, 0.0035) 6.74;
- **wall clock** at batch 16 inside ``lax.scan``, normalised per 0.007 time units, with
  both timing windows kept (CLAUDE.md trap 13a).
The cheapest setting that reaches a given tau is the bar a learned solver must clear.

Output: data/results/e10/frontier[_<tag>].json

Usage (guarded launcher; GPU, JAX env)
  scripts/safe_run.sh cfd_frontier env -u LD_LIBRARY_PATH -u CUDA_HOME PYTHONPATH=src \\
      .venvs/frost-mhd/bin/python experiments/e10_cfd/frontier.py
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
from w1_eval import RES, best_of, correlation_times  # noqa: E402

DT, REC = 0.007, 16


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", default="64,80,96,112,128")
    ap.add_argument("--dts", default="0.0035,0.007,0.014")
    ap.add_argument("--dealias", choices=("two_thirds", "exp"), default="two_thirds",
                    help="exp: the exponential filter instead of the 2/3 rule (spectral.Grid)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1024, help="timed length, in steps of 0.007")
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
    for n in (int(x) for x in args.sizes.split(",")):
        g = Grid(n, dealias=args.dealias)
        w0 = (w128 if g == g128 else spectral.truncate(g128, g, w128) if n < 128 else
              spectral.pad(g128, g, w128))  # above 128^2: the stored state, zero-padded
        wb = jnp.tile(w0, (-(-args.batch // w0.shape[0]), 1, 1))[: args.batch]
        for dt in (float(x) for x in args.dts.split(",")):
            per_record = REC * DT / dt
            if abs(per_record - round(per_record)) > 1e-6:
                raise ValueError(f"dt {dt} does not divide a record of {REC * DT}")
            per_record = round(per_record)
            states = schedule.rollout(g, None, w0, mode="none", k=1, dt=dt, n_records=n_rec,
                                      record_every=per_record)
            st64 = states if g == g64 else spectral.truncate(g, g64, states)  # scored on the 64^2 2/3 band
            rho = np.asarray(spectral.correlation(spectral.to_real(g64, st64), truth))
            tau = correlation_times(np.where(np.isfinite(rho), rho, -1.0), REC * DT)
            n_steps = round(args.steps * DT / dt)
            lo, hi = best_of(lambda: schedule.advance(g, None, wb, mode="none", k=1, dt=dt, n_outer=n_steps))
            name = f"spectral{n}{'exp' if args.dealias == 'exp' else ''}:dt{dt:g}"
            rows[name] = {"n": n, "dt": dt, "tau": tau.tolist(), "tau_mean": float(tau.mean()),
                          "tau_sem": float(tau.std(ddof=1) / np.sqrt(len(tau))),
                          "finite": bool(jnp.all(jnp.isfinite(states.real))),
                          "ms_per_0.007": 1e3 * lo / args.steps, "ms_per_0.007_slow_window": 1e3 * hi / args.steps}
            r = rows[name]
            print(f"{name:22s} tau {r['tau_mean']:5.2f} +- {r['tau_sem']:.2f} | finite {r['finite']} | "
                  f"{r['ms_per_0.007']:.3f} ms per 0.007 (windows {lo / hi:.2f})", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    out = RES / f"frontier{'_' + args.tag if args.tag else ''}.json"
    out.write_text(json.dumps({"batch": args.batch, "record_dt": REC * DT, "rows": rows}, indent=1))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
