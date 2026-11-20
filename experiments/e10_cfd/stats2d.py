"""Long-run statistics of W1 (Kolmogorov flow, Re 1000): classical solvers against the
split-operator hybrid, at every method's own cost.

tau measures forecast skill and is capped by the initial condition (``ceiling.py``: from
the 64^2 state, perfect dynamics reach 7.05, below filtered spectral 96^2). Statistics are
not: a long run forgets its initial state, so the question is how close each method's
time-averaged statistics come to the DNS, per unit cost.

Metrics, on the 64^2 2/3 band, averaged over 16 trajectories and over t in [t0, T]:
- spectrum: mean |log(E(k) / E_DNS(k))| over shells 1..21 (Kochkov's spectrum metric);
- energy and enstrophy: relative error of the means.
The DNS reference is the eval group's average over all records (16 trajectories x 15
time units, statistically stationary after the spin-up). Its own sampling noise is the
floor; classical 128^2 sits on it.

Output: data/results/e10/stats2d_<tag>.json

Usage (guarded launcher; GPU, JAX env)
  scripts/safe_run.sh cfd_stats env -u LD_LIBRARY_PATH -u CUDA_HOME PYTHONPATH=src \\
      .venvs/frost-mhd/bin/python experiments/e10_cfd/stats2d.py --tag base
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np

from frost.cfd import schedule, spectral, split
from frost.cfd.spectral import Grid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_closure import DATA  # noqa: E402
from w1_eval import RES  # noqa: E402
from w1split import load_split  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DT, REC = 0.007, 16
KB = 21  # shells scored: the 64^2 2/3 band


def stats(g23: Grid, st64, t0_rec: int):
    """Per trajectory: time-averaged spectrum (shells 1..KB), energy, enstrophy over records >= t0_rec."""
    st = st64[t0_rec:]
    spec = np.asarray(spectral.energy_spectrum(g23, st, kbins=KB)).mean(axis=0)[..., 1:]
    w = np.asarray(spectral.to_real(g23, st))
    enst = 0.5 * (w**2).mean(axis=(-2, -1)).mean(axis=0)
    return spec, spec.sum(-1), enst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--classical", default="64:0.014,64:0.028,80:0.014,96:0.014,112:0.0112,128:0.0112,128:0.007",
                    help="n:dt,... (filtered spectral)")
    ap.add_argument("--models", default="data/models/cfd/ksplit_k1:0.014:1,data/models/cfd/ksplit_k1:0.014:16",
                    help="path:dt:hold k,... (split-operator hybrids on 64^2)")
    ap.add_argument("--horizon", type=float, default=60.0, help="time units per run")
    ap.add_argument("--t0", type=float, default=5.0, help="statistics start here (the initial state forgotten)")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    g23, g128 = Grid(64), Grid(128)
    with h5py.File(args.data, "r") as fh:
        ev = np.asarray(fh["eval/w64"])
        ev128 = np.asarray(fh["eval/w128"])
    ref_hat = jnp.asarray(np.fft.rfft2(ev) * g23.mask, jnp.complex64)
    r_spec, r_e, r_z = stats(g23, ref_hat, 0)
    r_spec, r_e, r_z = r_spec.mean(0), r_e.mean(), r_z.mean()
    w128 = jnp.asarray(np.fft.rfft2(ev128[0]) * g128.mask, jnp.complex64)
    rec_t = REC * DT
    n_rec, t0_rec = round(args.horizon / rec_t), round(args.t0 / rec_t)
    print(f"reference: {ev.shape[0]} records x {ev.shape[1]} trajectories | E {r_e:.3f} | Z {r_z:.2f} | "
          f"runs {args.horizon} time units, statistics from t = {args.t0}", flush=True)

    rows = {}

    def score(name, st64, extra):
        finite = bool(jnp.all(jnp.isfinite(st64.real)))
        spec, e, z = stats(g23, st64, t0_rec) if finite else (None, None, None)
        if finite:
            l1 = np.abs(np.log(spec.mean(0) / r_spec)).mean()
            l1_each = np.abs(np.log(spec / r_spec)).mean(-1)
            row = {"spectrum_l1": float(l1), "spectrum_l1_per_traj_mean": float(l1_each.mean()),
                   "energy_err": float(e.mean() / r_e - 1), "enstrophy_err": float(z.mean() / r_z - 1),
                   "spectrum_ratio": (spec.mean(0) / r_spec).tolist()}
        else:
            row = {"spectrum_l1": float("nan")}
        rows[name] = {"finite": finite, **row, **extra}
        if finite:
            print(f"{name:28s} spectrum L1 {row['spectrum_l1']:.4f} (per trajectory {row['spectrum_l1_per_traj_mean']:.4f})"
                  f" | E {100 * row['energy_err']:+.1f}% | Z {100 * row['enstrophy_err']:+.1f}%", flush=True)
        else:
            print(f"{name:28s} non-finite", flush=True)

    for c in (x for x in args.classical.split(",") if x):
        n, cdt = int(c.split(":")[0]), float(c.split(":")[1])
        g = Grid(n, dealias="exp")
        w0 = spectral.truncate(g128, g, w128) if n <= 128 else spectral.pad(g128, g, w128)
        per = round(rec_t / cdt)
        st = schedule.rollout(g, None, w0, mode="none", k=1, dt=cdt, n_records=n_rec, record_every=per)
        score(f"spectral{n}exp:dt{cdt:g}", spectral.truncate(g, g23, st), {"n": n, "dt": cdt})
    g64 = Grid(64, dealias="exp")
    w0_64 = spectral.truncate(g128, g64, w128)
    for mdef in (x for x in args.models.split(",") if x):
        path, mdt, k = mdef.split(":")
        mdt, k = float(mdt), int(k)
        model = load_split(ROOT / path)
        per = round(rec_t / mdt)
        st = split.rollout(g64, model, w0_64, mode="value", k=k, dt=mdt, n_records=n_rec, record_every=per)
        score(f"split:{Path(path).name}:dt{mdt:g}:k{k}", spectral.truncate(g64, g23, st),
              {"model": path, "dt": mdt, "k": k})

    out = RES / f"stats2d_{args.tag}.json"
    out.write_text(json.dumps({"horizon": args.horizon, "t0": args.t0, "rows": rows,
                               "reference_spectrum": r_spec.tolist()}, indent=1))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
