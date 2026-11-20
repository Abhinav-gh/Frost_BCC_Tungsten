"""Reference DNS for workload W1: Kolmogorov flow at Re = 1000 (new_directions.md S3, test 11).

The ground truth every coarse model is scored against, as in Kochkov et al. 2021:
- DNS on 1024^2 with our IFRK4 pseudo-spectral solver (``frost.cfd.spectral``);
- dt_fine = dt_64 / 16, with dt_64 = 0.007 (CFL 0.5 at |u| = 7 on 64^2, Kochkov's
  choice);
- random initial fields spun up for ``--spinup`` time units to the statistically
  steady state;
- sharp spectral filtering onto each coarse grid's 2/3 band.

Two groups, from different keys:
- **train**: 8 trajectories over ``--train-T``, the 64^2 field every 64^2 step (the
  unrolled-training target) and the 128^2 field every 8 of those;
- **eval**: 16 trajectories over ``--eval-T``, both fields every 16 64^2-steps
  (0.112 time units), for correlation times and spectra.

The fine-grid energy spectrum is stored alongside, for resolution checks.

Output: data/cfd/kolmogorov_re1000.h5

Usage (guarded launcher; GPU, JAX env)
  scripts/safe_run.sh cfd_dns env -u LD_LIBRARY_PATH -u CUDA_HOME PYTHONPATH=src \\
      .venvs/frost-mhd/bin/python experiments/e10_cfd/kolmogorov_dns.py
"""

from __future__ import annotations

import argparse
import time
from functools import partial
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from frost.cfd import spectral
from frost.cfd.spectral import Grid

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "cfd" / "kolmogorov_re1000.h5"
DT64 = 0.007
SUB = 16  # fine steps per 64^2 step


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-fine", type=int, default=1024)
    ap.add_argument("--spinup", type=float, default=25.0)
    ap.add_argument("--train-T", type=float, default=20.0)
    ap.add_argument("--eval-T", type=float, default=15.0)
    ap.add_argument("--n-train", type=int, default=8)
    ap.add_argument("--n-eval", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=64, help="64^2 steps per compiled call")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--train-key", type=int, default=0, help="PRNG key of the training initial conditions")
    ap.add_argument("--train-only", action="store_true",
                    help="write only a train group (extra training data, a new --train-key, a new --out)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    out = args.out
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)

    fine, g64, g128 = Grid(args.n_fine), Grid(64), Grid(128)
    dt = DT64 / SUB
    print(f"devices {jax.devices()} | fine {fine.n}^2 dt {dt:.3e} | 64^2 dt {DT64}", flush=True)

    @partial(jax.jit, static_argnames=("n",))
    def advance(w, n: int):
        return jax.lax.fori_loop(0, n, lambda _, x: spectral.step(fine, x, dt), w)

    @partial(jax.jit, static_argnames=("n_snap", "every"))
    def snapshots(w, n_snap: int, every: int):
        """n_snap filtered snapshots, ``every`` 64^2-steps apart."""

        def one(x, _):
            x = jax.lax.fori_loop(0, SUB * every, lambda _, y: spectral.step(fine, y, dt), x)
            f64 = spectral.to_real(g64, spectral.truncate(fine, g64, x))
            f128 = spectral.to_real(g128, spectral.truncate(fine, g128, x))
            return x, (f64, f128, spectral.energy_spectrum(fine, x, kbins=fine.kmax),
                       spectral.max_speed(fine, x))

        return jax.lax.scan(one, w, None, length=n_snap)

    with h5py.File(out, "w") as fh:
        fh.attrs.update({"n_fine": fine.n, "dt_fine": dt, "dt64": DT64, "dt128": DT64 / 2,
                         "nu": fine.nu, "alpha": fine.alpha, "k_force": fine.k_force,
                         "spinup": args.spinup, "solver": "frost.cfd.spectral IFRK4, 2/3 rule"})
        if args.train_key == 1:
            raise SystemExit("--train-key 1 is the eval key: training on it would leak the eval set")
        groups = [("train", args.train_key, args.n_train, args.train_T, 1, 8)]
        if not args.train_only:
            groups.append(("eval", 1, args.n_eval, args.eval_T, 16, 16))
        for group, key, batch, horizon, every64, every128 in groups:
            w = spectral.random_initial(fine, jax.random.PRNGKey(key), batch=(batch,))
            n_spin = round(args.spinup / dt)
            t0 = time.perf_counter()
            done = 0
            while done < n_spin:
                m = min(SUB * args.chunk, n_spin - done)
                w = advance(w, m)
                done += m
                if done == m:
                    w.block_until_ready()
                    print(f"{group}: {1e3 * (time.perf_counter() - t0) / m:.2f} ms per fine step "
                          f"(batch {batch}, incl. compile)", flush=True)
            w.block_until_ready()
            print(f"{group}: spun up {args.spinup} in {time.perf_counter() - t0:.0f} s | max |u| "
                  f"{float(spectral.max_speed(fine, w)):.2f}", flush=True)

            n64 = round(horizon / DT64) // every64
            d64 = fh.create_dataset(f"{group}/w64", (n64, batch, 64, 64), "f4")
            d128 = fh.create_dataset(f"{group}/w128", (n64 * every64 // every128, batch, 128, 128), "f4")
            dspec = fh.create_dataset(f"{group}/spectrum_fine", (n64 * every64 // every128, batch,
                                                                 fine.kmax + 1), "f4")
            fh[group].attrs.update({"every64_steps": every64, "every128_in_64_steps": every128,
                                    "batch": batch, "key": key})
            written, written128, umax = 0, 0, 0.0
            t1 = time.perf_counter()
            while written < n64:
                m = min(args.chunk if every64 == 1 else max(args.chunk // every64, 1), n64 - written)
                w, (f64, f128, spec, speed) = snapshots(w, m, every64)
                f64, f128, spec = np.asarray(f64), np.asarray(f128), np.asarray(spec)
                d64[written:written + m] = f64
                stride = every128 // every64
                for i in range(m):
                    if (written + i + 1) % stride == 0 and written128 < d128.shape[0]:
                        d128[written128] = f128[i]
                        dspec[written128] = spec[i]
                        written128 += 1
                written += m
                umax = max(umax, float(np.max(speed)))
                if not np.all(np.isfinite(f64)):
                    raise SystemExit(f"{group}: non-finite field at snapshot {written}")
            print(f"{group}: {n64} snapshots in {time.perf_counter() - t1:.0f} s | max |u| over "
                  f"window {umax:.2f} (CFL at 64^2: {umax * DT64 / g64.dx:.2f})", flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
