"""The closure on the second GPU, one outer step behind, on workload W1 (test 13).

Each configuration ``mode:k:lag:devices`` runs ``frost.cfd.pipeline.run``:
- ``lag 0, 1 device`` is the Python-driven loop on one GPU. It isolates dispatch
  cost against the ``lax.scan`` schedules of test 12.
- ``lag 0, 2 devices`` adds the transfers but no overlap.
- ``lag 1, 2 devices`` is the pipeline.

Accuracy is scored as in ``w1_eval.py`` (correlation time against the filtered DNS
over the 16 eval trajectories, paired against B0 = the ``lax.scan`` every-step
operator schedule). Wall clock is the best of three windows of ``--steps`` solver
steps at each batch size.

Output: data/results/e10/w1_async_<tag>.json

Usage (guarded launcher; both GPUs, JAX env)
  scripts/safe_run.sh cfd_w1_async env -u LD_LIBRARY_PATH -u CUDA_HOME PYTHONPATH=src \\
      .venvs/frost-mhd/bin/python experiments/e10_cfd/w1_async.py --tag test13
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from frost.cfd import pipeline, schedule, spectral
from frost.cfd.spectral import Grid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_closure import DATA, OUT as MODEL, load_model  # noqa: E402
from w1_eval import DT64, REC, RES, correlation_times  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", default=("operator:1:0:1,operator:1:0:2,operator:1:1:2,"
                                          "operator:2:1:2,operator:4:1:2,operator:8:1:2,"
                                          "value:4:1:2,value:8:1:2"))
    ap.add_argument("--batches", default="1,16")
    ap.add_argument("--steps", type=int, default=1024)
    ap.add_argument("--model", type=Path, default=MODEL)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    devs = jax.devices()
    if len(devs) < 2:
        raise SystemExit(f"needs two devices, have {devs}")
    d0, d1 = devs[0], devs[1]
    g = Grid(64)
    model = load_model(args.model)
    with h5py.File(args.data, "r") as fh:
        ev = np.asarray(fh["eval/w64"])
    n_rec = ev.shape[0] - 1
    w0 = jax.device_put(jnp.asarray(np.fft.rfft2(ev[0]) * g.mask, jnp.complex64), d0)
    truth = jnp.asarray(ev[1:])

    def score(states):
        rho = np.asarray(spectral.correlation(spectral.to_real(g, states), truth))
        return correlation_times(np.where(np.isfinite(rho), rho, -1.0), REC * DT64)

    b0_states = schedule.rollout(g, jax.device_put(model, d0), w0, mode="operator", k=1, dt=DT64,
                                 n_records=n_rec, record_every=REC)
    tau_b0 = score(b0_states)
    print(f"B0 (scan, operator:1) tau {tau_b0.mean():.2f} +- "
          f"{tau_b0.std(ddof=1) / np.sqrt(len(tau_b0)):.2f}", flush=True)

    rows = {}
    for cfg in args.configs.split(","):
        mode, k, lag, nd = cfg.split(":")
        k, lag, nd = int(k), int(lag), int(nd)
        dc = d1 if nd == 2 else d0
        _, recs = pipeline.run(g, model, w0, mode=mode, k=k, dt=DT64, n_outer=n_rec * REC // k,
                               lag=lag, solver_device=d0, closure_device=dc, record_every=REC)
        tau = score(jnp.stack([jax.device_put(r, d0) for r in recs]))
        d = tau - tau_b0
        row = {"mode": mode, "k": k, "lag": lag, "devices": nd, "tau_mean": float(tau.mean()),
               "dtau_vs_B0": float(d.mean()), "dtau_sem": float(d.std(ddof=1) / np.sqrt(len(d))),
               "timing_ms_per_step": {}}
        row["matched_accuracy"] = bool(row["dtau_vs_B0"] >= -2 * row["dtau_sem"])
        for b in (int(x) for x in args.batches.split(",")):
            wb = jnp.tile(w0, (-(-b // w0.shape[0]), 1, 1))[:b]

            def go():
                w, _ = pipeline.run(g, model, wb, mode=mode, k=k, dt=DT64, n_outer=args.steps // k,
                                    lag=lag, solver_device=d0, closure_device=dc)
                return w

            go().block_until_ready()
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                go().block_until_ready()
                times.append(time.perf_counter() - t0)
            row["timing_ms_per_step"][str(b)] = 1e3 * min(times) / args.steps
            row.setdefault("window_spread", {})[str(b)] = max(times) / min(times) - 1
        rows[cfg] = row
        print(f"{cfg:18s} tau {row['tau_mean']:5.2f} (vs B0 {row['dtau_vs_B0']:+.2f} +- "
              f"{row['dtau_sem']:.2f}) | ms/step " + " ".join(
                  f"b{b} {v:.3f}" for b, v in row["timing_ms_per_step"].items()), flush=True)

    RES.mkdir(parents=True, exist_ok=True)
    out = RES / f"w1_async_{args.tag}.json"
    out.write_text(json.dumps({"B0_tau": tau_b0.tolist(), "rows": rows,
                               "devices": [str(d) for d in devs]}, indent=1))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
