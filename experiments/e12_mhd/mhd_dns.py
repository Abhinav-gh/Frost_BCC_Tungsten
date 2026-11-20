"""Reference data for W5: forced incompressible MHD turbulence (magnetic Prandtl 1), DNS on
n_fine^3 with ``frost.cfd.mhd3d``, filtered to the coarse grids every coarse step.

The W3 recipe (``experiments/e10_cfd/hit_dns.py``) with both fields: constant power into |k| <= 2
of the flow (``--power``) and, optionally, of the field (``--power-b``; default 0, dynamo-sustained), random initial fields, spun up ``--spinup`` time
units before anything is recorded. Stored, per trajectory, every coarse step (0.04): the state's
six components (u, b) on each ``--coarse`` grid's band as real fields ``u<n>`` (T, B, 6, n, n, n),
float32, under the same keys as W3 so the trainer and the statistics read either; the evaluation
group also keeps the DNS state at t = 0 truncated to min(128, n_fine)^3 (``eval/u_ic``). ``stats``
per record: kinetic energy, magnetic energy, total dissipation, cross helicity, max |u| + max |b|.

Output: data/cfd/<out>.h5 (train/u<n>, eval/u<n>, eval/u_ic, */stats, attrs incl. physics = "mhd")

Usage (guarded launcher; GPU, JAX env)
  scripts/safe_run.sh mhd_dns env -u LD_LIBRARY_PATH -u CUDA_HOME PYTHONPATH=src \\
      .venvs/frost-mhd/bin/python experiments/e12_mhd/mhd_dns.py --out data/cfd/mhd256.h5
"""

from __future__ import annotations

import argparse
import time
from functools import partial
from pathlib import Path

import h5py
import jax
import numpy as np

from frost.cfd import mhd3d, spectral3d as s3
from frost.cfd.spectral3d import Grid3

ROOT = Path(__file__).resolve().parents[2]
DT_C = 0.04


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-fine", type=int, default=256)
    ap.add_argument("--sub", type=int, default=8, help="fine steps per coarse step")
    ap.add_argument("--nu", type=float, default=1.5e-3, help="viscosity = magnetic diffusivity")
    ap.add_argument("--power", type=float, default=0.1, help="power injected into the flow")
    ap.add_argument("--power-b", type=float, default=0.0,
                    help="power injected into the field (0: flow-only forcing, the field sustained by dynamo action)")
    ap.add_argument("--spinup", type=float, default=10.0)
    ap.add_argument("--train-T", type=float, default=12.0)
    ap.add_argument("--eval-T", type=float, default=2.0)
    ap.add_argument("--n-train", type=int, default=6)
    ap.add_argument("--n-eval", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=25, help="coarse steps per compiled call")
    ap.add_argument("--coarse", default="32,48")
    ap.add_argument("--save-state", type=Path, default=None,
                    help="directory: write each group's final fine state (<group>_n<n>.npy) for a later --init-from")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="directory of --save-state output from a coarser run: each group starts from its state, "
                         "zero-padded to this grid (progressive spin-up: the large scales equilibrate on the cheap "
                         "grid, --spinup here only fills the new small scales)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=0,
                    help="independent random initial fields (keys 2 seed and 2 seed + 1): a fresh reference never used "
                         "for selection")
    args = ap.parse_args()
    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"{args.out} exists; pass --overwrite")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fine = Grid3(args.n_fine, nu=args.nu, power=args.power, power_b=args.power_b)
    coarse = [Grid3(int(n), nu=args.nu, power=args.power, power_b=args.power_b, dealias="exp")
              for n in args.coarse.split(",")]
    n_ic = min(128, args.n_fine)
    g_ic = Grid3(n_ic, nu=args.nu, power=args.power, power_b=args.power_b, dealias="exp")
    dt = DT_C / args.sub
    print(f"devices {jax.devices()} | MHD fine {fine.n}^3 dt {dt:.4f} | nu = eta {args.nu} | power {args.power} "
          f"per field | coarse step {DT_C}", flush=True)

    @partial(jax.jit, static_argnames=("n",))
    def advance(s, n: int):
        return jax.lax.fori_loop(0, n, lambda _, x: mhd3d.step(fine, x, dt), s)

    def stats_of(s):
        ek, em = mhd3d.energies(fine, s)
        return ek, em, mhd3d.dissipation(fine, s), mhd3d.cross_helicity(fine, s), mhd3d.max_signal_speed(fine, s)

    @partial(jax.jit, static_argnames=("m",))
    def snapshots(s, m: int):
        def one(x, _):
            x = jax.lax.fori_loop(0, args.sub, lambda _, y: mhd3d.step(fine, y, dt), x)
            return x, (tuple(s3.to_real(g, s3.truncate(fine, g, x)) for g in coarse), stats_of(x))

        return jax.lax.scan(one, s, None, length=m)

    with h5py.File(args.out, "w") as fh:
        fh.attrs.update({"physics": "mhd", "n_fine": fine.n, "dt_fine": dt, "dt_coarse": DT_C, "nu": args.nu,
                         "eta": args.nu, "power": args.power, "power_b": args.power_b, "k_force": fine.k_force, "spinup": args.spinup,
                         "n_ic": n_ic, "solver": "frost.cfd.mhd3d IFRK4, 2/3 rule"})
        for group, key, batch, horizon in (("train", 0, args.n_train, args.train_T), ("eval", 1, args.n_eval, args.eval_T)):
            if args.init_from is not None:
                src = next(args.init_from.glob(f"{group}_n*.npy"))
                n_src = int(src.stem.split("_n")[1])
                s = s3.pad(Grid3(n_src, nu=args.nu, power=args.power, power_b=args.power_b), fine,
                           jax.numpy.asarray(np.load(src)))
                if s.shape[0] != batch:
                    raise SystemExit(f"{src} holds {s.shape[0]} trajectories, --n-{group} is {batch}")
                print(f"{group}: start from {src.name} ({n_src}^3, padded)", flush=True)
            else:
                s = mhd3d.random_initial(fine, jax.random.PRNGKey(key + 2 * args.seed), batch=(batch,))
            t0 = time.perf_counter()
            s = advance(s, 20)
            s.block_until_ready()
            print(f"{group}: {1e3 * (time.perf_counter() - t0) / 20:.1f} ms per fine step (batch {batch}, incl. compile)",
                  flush=True)
            n_spin, done = round(args.spinup / dt), 20
            while done < n_spin:
                m = min(400, n_spin - done)
                s = advance(s, m)
                done += m
            s.block_until_ready()
            ek, em, eps, hc, vmax = (np.asarray(x) for x in stats_of(s))
            u_rms = np.sqrt(2 * ek / 3)
            lam = np.sqrt(15 * fine.nu * u_rms**2 / np.maximum(eps, 1e-12))
            eta_k = (fine.nu**3 / np.maximum(eps, 1e-12)) ** 0.25
            print(f"{group}: spun up {args.spinup} in {time.perf_counter() - t0:.0f} s | E_kin {ek.mean():.3f} "
                  f"E_mag {em.mean():.3f} | eps {eps.mean():.3f} | cross helicity {hc.mean():+.3f} | Re_lambda "
                  f"{(u_rms * lam / fine.nu).mean():.0f} | k_max eta {(fine.kmax * eta_k).mean():.2f} | "
                  f"max |u| + |b| {float(vmax):.2f}", flush=True)
            if group == "eval":
                fh.create_dataset("eval/u_ic", data=np.asarray(s3.to_real(g_ic, s3.truncate(fine, g_ic, s))))
            n_c = round(horizon / DT_C)
            dsets = [fh.create_dataset(f"{group}/u{g.n}", (n_c, batch, 6, g.n, g.n, g.n), "f4") for g in coarse]
            stats, written = [], 0
            while written < n_c:
                m = min(args.chunk, n_c - written)
                s, (flds, (ek, em, eps, hc, vmax)) = snapshots(s, m)
                for d, f in zip(dsets, flds, strict=True):
                    d[written:written + m] = np.asarray(f)
                stats.append(np.stack([np.asarray(ek).mean(-1), np.asarray(em).mean(-1), np.asarray(eps).mean(-1),
                                       np.asarray(hc).mean(-1), np.asarray(vmax)], -1))
                written += m
            fh.create_dataset(f"{group}/stats", data=np.concatenate(stats))  # (T, 5)
            if args.save_state is not None:
                args.save_state.mkdir(parents=True, exist_ok=True)
                np.save(args.save_state / f"{group}_n{fine.n}.npy", np.asarray(s))
            print(f"{group}: {n_c} coarse steps written in {time.perf_counter() - t0:.0f} s", flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
