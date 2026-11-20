"""Accuracy and wall clock of every schedule on workload W1 (tests 11-12, new_directions.md S4).

For each configuration ``mode:k`` (``frost.cfd.schedule.MODES``):
- **accuracy** against the filtered DNS over the 16 eval trajectories: the vorticity
  correlation time (Kochkov's metric, time until rho < 0.95, at 0.112 resolution) and
  the time-averaged energy-spectrum L1. Each is also reported as a *paired*
  difference against B0 (``operator:1``, the ML solver's every-step form) with its
  standard error. A schedule is matched-accuracy if the paired correlation-time
  difference is within two standard errors of zero, or better.
- **wall clock**: ``advance`` over ``--steps`` solver steps inside ``lax.scan``, best
  of three windows after compilation, at each batch size in ``--batches``. Also
  timed: the CNN call alone and the solver alone, which give the closure's share
  of a B0 step.

The direct simulation on 128^2 (``none128``, from the 128^2-filtered DNS) is the
resolution baseline. Its error is measured after truncation to the 64^2 band.

Output: data/results/e10/w1_eval_<tag>.json

Usage (guarded launcher; GPU, JAX env)
  scripts/safe_run.sh cfd_w1_eval env -u LD_LIBRARY_PATH -u CUDA_HOME PYTHONPATH=src \\
      .venvs/frost-mhd/bin/python experiments/e10_cfd/w1_eval.py --tag test11
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

from frost.cfd import closure, schedule, spectral
from frost.cfd.spectral import Grid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_closure import DATA, OUT as MODEL, load_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "data" / "results" / "e10"
DT64, REC = 0.007, 16  # one record = 16 steps on 64^2 = 0.112 time units


def correlation_times(rho: np.ndarray, dt_record: float) -> np.ndarray:
    """Per trajectory: time of the first record with rho < 0.95 (the window end if never)."""
    below = rho < 0.95
    first = np.where(below.any(axis=0), below.argmax(axis=0), rho.shape[0])
    return (first + 1) * dt_record


def spectrum_l1(g: Grid, states_hat, ref_hat) -> np.ndarray:
    """Per trajectory: L1 of the time-averaged spectrum, relative, over shells 1..kmax."""
    a = np.asarray(spectral.energy_spectrum(g, states_hat, kbins=g.kmax)).mean(axis=0)[..., 1:]
    b = np.asarray(spectral.energy_spectrum(g, ref_hat, kbins=g.kmax)).mean(axis=0)[..., 1:]
    return np.abs(np.log(a / b)).mean(axis=-1)  # mean |log ratio| per shell: scale-free


def best_of(fn, windows: int = 3) -> float:
    fn().block_until_ready()  # compile and warm
    times = []
    for _ in range(windows):
        t0 = time.perf_counter()
        fn().block_until_ready()
        times.append(time.perf_counter() - t0)
    return min(times), max(times)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", default="none:1,smag:1,operator:1,value:1,none128:1")
    ap.add_argument("--batches", default="1,16")
    ap.add_argument("--steps", type=int, default=1024, help="solver steps per timing window")
    ap.add_argument("--cs", type=float, default=0.17)
    ap.add_argument("--model", type=Path, default=MODEL)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    g64, g128 = Grid(64), Grid(128)
    model = load_model(args.model)
    with h5py.File(args.data, "r") as fh:
        ev64 = np.asarray(fh["eval/w64"])    # (R, B, 64, 64)
        ev128 = np.asarray(fh["eval/w128"])  # (R, B, 128, 128)
    n_rec = ev64.shape[0] - 1
    ref_hat = jnp.asarray(np.fft.rfft2(ev64[1:]) * g64.mask, jnp.complex64)
    w0_64 = jnp.asarray(np.fft.rfft2(ev64[0]) * g64.mask, jnp.complex64)
    w0_128 = jnp.asarray(np.fft.rfft2(ev128[0]) * g128.mask, jnp.complex64)
    configs = [(c.split(":")[0], int(c.split(":")[1])) for c in args.configs.split(",")]
    batches = [int(b) for b in args.batches.split(",")]
    print(f"devices {jax.devices()} | eval {ev64.shape} | {n_rec} records of {REC * DT64:.3f} | "
          f"CNN {closure.cnn_flops(model['params'], 64) / 1e9:.2f} GFLOP per call", flush=True)

    rows: dict[str, dict] = {}
    for mode, k in configs:
        name = f"{mode}:{k}"
        if mode == "none128":
            states = schedule.rollout(g128, None, w0_128, mode="none", k=1, dt=DT64 / 2,
                                      n_records=n_rec, record_every=2 * REC)
            states = spectral.truncate(g128, g64, states)
        else:
            states = schedule.rollout(g64, model, w0_64, mode=mode, k=k, dt=DT64, n_records=n_rec,
                                      record_every=REC, cs=args.cs)
        finite = bool(jnp.all(jnp.isfinite(states.real)))
        rho = np.asarray(spectral.correlation(spectral.to_real(g64, states), jnp.asarray(ev64[1:])))
        rho = np.where(np.isfinite(rho), rho, -1.0)
        tau = correlation_times(rho, REC * DT64)
        spec = spectrum_l1(g64, states, ref_hat) if finite else np.full(tau.shape, np.nan)
        rows[name] = {"mode": mode, "k": k, "finite": finite, "tau": tau.tolist(),
                      "tau_mean": float(tau.mean()), "tau_sem": float(tau.std(ddof=1) / np.sqrt(len(tau))),
                      "spectrum_l1": spec.tolist(), "spectrum_l1_mean": float(np.mean(spec))}
        print(f"{name:16s} tau {tau.mean():5.2f} +- {rows[name]['tau_sem']:.2f} | spectrum "
              f"|log ratio| {np.mean(spec):.3f} | finite {finite}", flush=True)

    base = rows.get("operator:1")
    if base is not None:
        for name, r in rows.items():
            d = np.asarray(r["tau"]) - np.asarray(base["tau"])
            ds = np.asarray(r["spectrum_l1"]) - np.asarray(base["spectrum_l1"])
            sem = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
            r["paired_vs_B0"] = {"dtau": float(d.mean()), "dtau_sem": sem,
                                 "dspec": float(ds.mean()),
                                 "dspec_sem": float(ds.std(ddof=1) / np.sqrt(len(ds)))}
            r["matched_accuracy"] = bool(d.mean() >= -2 * sem) if sem == sem else None

    timing: dict[str, dict] = {}
    for b in batches:
        reps = -(-b // w0_64.shape[0])
        w64 = jnp.tile(w0_64, (reps, 1, 1))[:b]
        w128 = jnp.tile(w0_128, (reps, 1, 1))[:b]
        t: dict[str, float] = {}
        coef = jax.jit(lambda w: closure.coefficients(g64, model, w))
        lo, hi = best_of(lambda: coef(w64))
        t["cnn_call_ms"] = 1e3 * lo
        for mode, k in configs:
            name = f"{mode}:{k}"
            if mode == "none128":
                lo, hi = best_of(lambda: schedule.advance(g128, None, w128, mode="none", k=1,
                                                          dt=DT64 / 2, n_outer=2 * args.steps))
                # per 64^2-equivalent step of physical time
            else:
                lo, hi = best_of(lambda: schedule.advance(g64, model, w64, mode=mode, k=k, dt=DT64,
                                                          n_outer=args.steps // k, cs=args.cs))
            t[name] = 1e3 * lo / args.steps
            t[name + "_window_spread"] = hi / lo - 1.0
        b0 = t.get("operator:1")
        speed = {n: b0 / v for n, v in t.items() if b0 and not n.endswith("spread") and n != "cnn_call_ms"}
        share = ((t["operator:1"] - t["none:1"]) / t["operator:1"]
                 if "operator:1" in t and "none:1" in t else None)
        timing[str(b)] = {"ms_per_step": t, "speedup_vs_B0": speed, "closure_share_of_B0": share}
        print(f"batch {b}: " + " | ".join(f"{n} {v:.3f}" for n, v in t.items()
                                         if not n.endswith("spread")) + " ms", flush=True)
        print(f"batch {b}: closure share of a B0 step "
              f"{share if share is None else round(share, 3)} | speedups "
              + " ".join(f"{n} {v:.2f}x" for n, v in speed.items()), flush=True)

    RES.mkdir(parents=True, exist_ok=True)
    out = RES / f"w1_eval_{args.tag}.json"
    out.write_text(json.dumps({"configs": args.configs, "cs": args.cs, "model": str(args.model),
                               "records": n_rec, "record_dt": REC * DT64, "accuracy": rows,
                               "timing": timing, "devices": [str(d) for d in jax.devices()]},
                              indent=1))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
