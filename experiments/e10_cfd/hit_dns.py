"""Reference data for W3: forced isotropic turbulence, DNS on n_fine^3, filtered to the
coarse grid every coarse step.

- DNS: ``frost.cfd.spectral3d`` at 192^3 (2/3 rule, k_max 64), nu 3e-3, constant power 0.1
  into |k| <= 2: k_max eta ~ 1.5 (resolved). Random initial fields, spun up 10 time
  units (~5 large-eddy turnovers) before anything is recorded.
- Stored, per trajectory, every coarse step (0.04 time units): the velocity on each
  ``--coarse`` grid's full band (32^3: |k| <= 15), as real fields u<n> (T, B, 3, n, n, n),
  float32. u32 is the scoring band; each is a training target.
- Eval trajectories also keep the DNS state at t = 0 truncated to min(128, n_fine)^3
  (attr ``n_ic``), the initial condition for every classical comparison (each method
  starts from the best state its grid can hold).

Output: data/cfd/hit192.h5 (train/u32, eval/u32, eval/u_ic, attrs)

Usage (guarded launcher; GPU, JAX env)
  scripts/safe_run.sh hit_dns env -u LD_LIBRARY_PATH -u CUDA_HOME PYTHONPATH=src \\
      .venvs/frost-mhd/bin/python experiments/e10_cfd/hit_dns.py
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

from frost.cfd import spectral3d as s3
from frost.cfd.spectral3d import Grid3

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "cfd" / "hit192.h5"
DT_C = 0.04  # coarse (32^3) step


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-fine", type=int, default=192)
    ap.add_argument("--sub", type=int, default=6, help="fine steps per coarse step")
    ap.add_argument("--nu", type=float, default=3e-3)
    ap.add_argument("--power", type=float, default=0.1)
    ap.add_argument("--spinup", type=float, default=10.0)
    ap.add_argument("--train-T", type=float, default=20.0)
    ap.add_argument("--eval-T", type=float, default=6.0)
    ap.add_argument("--n-train", type=int, default=8)
    ap.add_argument("--n-eval", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=25, help="coarse steps per compiled call")
    ap.add_argument("--coarse", default="32", help="grids whose band is stored every coarse step (u<n>)")
    ap.add_argument("--groups", default="train,eval")
    ap.add_argument("--train-key", type=int, default=0, help="PRNG key of the training initial states (1 is eval)")
    ap.add_argument("--eval-key", type=int, default=1, help="PRNG key of the evaluation starts (a fresh reference: new keys)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"{args.out} exists; pass --overwrite")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fine = Grid3(args.n_fine, nu=args.nu, power=args.power)
    coarse = [Grid3(int(n), nu=args.nu, power=args.power, dealias="exp") for n in args.coarse.split(",")]
    if args.train_key == 1:
        raise SystemExit("--train-key 1 is the eval key")
    n_ic = min(128, args.n_fine)
    g_ic = Grid3(n_ic, nu=args.nu, power=args.power, dealias="exp")
    dt = DT_C / args.sub
    print(f"devices {jax.devices()} | fine {fine.n}^3 dt {dt:.4f} | coarse step {DT_C}", flush=True)

    @partial(jax.jit, static_argnames=("n",))
    def advance(u, n: int):
        return jax.lax.fori_loop(0, n, lambda _, x: s3.step(fine, x, dt), u)

    @partial(jax.jit, static_argnames=("m",))
    def snapshots(u, m: int):
        def one(x, _):
            x = jax.lax.fori_loop(0, args.sub, lambda _, y: s3.step(fine, y, dt), x)
            return x, (tuple(s3.to_real(g, s3.truncate(fine, g, x)) for g in coarse), s3.energy_of(fine, x),
                       s3.dissipation(fine, x), s3.max_speed(fine, x))

        return jax.lax.scan(one, u, None, length=m)

    with h5py.File(args.out, "w") as fh:
        fh.attrs.update({"n_fine": fine.n, "dt_fine": dt, "dt_coarse": DT_C, "nu": args.nu,
                         "power": args.power, "k_force": fine.k_force, "spinup": args.spinup, "n_ic": n_ic,
                         "solver": "frost.cfd.spectral3d IFRK4, 2/3 rule"})
        for group, key, batch, horizon in (("train", args.train_key, args.n_train, args.train_T),
                                           ("eval", args.eval_key, args.n_eval, args.eval_T)):
            if group not in args.groups.split(","):
                continue
            u = s3.random_initial(fine, jax.random.PRNGKey(key), batch=(batch,))
            t0 = time.perf_counter()
            n_spin = round(args.spinup / dt)
            u = advance(u, 50)
            u.block_until_ready()
            print(f"{group}: {1e3 * (time.perf_counter() - t0) / 50:.1f} ms per fine step (batch {batch}, "
                  f"incl. compile)", flush=True)
            done = 50
            while done < n_spin:
                m = min(500, n_spin - done)
                u = advance(u, m)
                done += m
            u.block_until_ready()
            e, eps = s3.energy_of(fine, u), s3.dissipation(fine, u)
            u_rms = jnp.sqrt(2 * e / 3)
            lam = jnp.sqrt(15 * fine.nu * u_rms**2 / eps)
            eta = (fine.nu**3 / eps) ** 0.25
            print(f"{group}: spun up {args.spinup} in {time.perf_counter() - t0:.0f} s | E {float(e.mean()):.3f} "
                  f"eps {float(eps.mean()):.3f} | Re_lambda {float((u_rms * lam / fine.nu).mean()):.0f} | "
                  f"k_max eta {float((fine.kmax * eta).mean()):.2f} | max |u| {float(s3.max_speed(fine, u)):.2f}",
                  flush=True)
            if group == "eval":
                fh.create_dataset("eval/u_ic", data=np.asarray(s3.to_real(g_ic, s3.truncate(fine, g_ic, u))))
            n_c = round(horizon / DT_C)
            dsets = [fh.create_dataset(f"{group}/u{g.n}", (n_c, batch, 3, g.n, g.n, g.n), "f4") for g in coarse]
            stats = []
            written = 0
            while written < n_c:
                m = min(args.chunk, n_c - written)
                u, (fields, e, eps, vmax) = snapshots(u, m)
                for d, f in zip(dsets, fields):
                    d[written:written + m] = np.asarray(f)
                stats.append(np.stack([np.asarray(e).mean(-1), np.asarray(eps).mean(-1), np.asarray(vmax)], -1))
                written += m
            fh.create_dataset(f"{group}/stats", data=np.concatenate(stats))  # (T, 3): E, eps, max |u|
            print(f"{group}: {n_c} coarse steps written in {time.perf_counter() - t0:.0f} s", flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
