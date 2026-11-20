"""W1 on the spectral solver with a learned correction: train, score, time (CFD accuracy push).

The finite-volume learned interpolation plateaued near tau 4.3 at 64^2, against spectral
DS at 128^2 (tau 6.74). At 64^2 spectral and finite-volume DS score alike (2.48 against
2.50): the gap is missing subgrid physics, not numerics. Dresdner et al. (2022) report
learned corrections to *spectral* solvers 2-4x more accurate than the spectral solver at
the same resolution, at about twice its runtime. So this runs:
- **base**: our IFRK4 pseudo-spectral solver on 64^2 (``frost.cfd.spectral``);
- **correction**: a CNN giving a bounded local flux operator, tendency -div(sum_s a_s omega(x+s))
  (``frost.cfd.closure``, stencil form). The divergence form conserves mean vorticity;
  the operator is held and applied to the fresh state (operator-consumed);
- **training**: through the solver on unrolls against the filtered DNS, with a warmup,
  updates skipped when non-finite, a stuck-run guard, and optionally the hold inside
  the unroll (schedule-aware, ``--train-k``);
- **scores**: tau over the 16 eval trajectories for DS 64^2, DS 128^2, Smagorinsky, the
  corrected solver every step (B0), plain holds, and certified intervals (the hold
  error checked at each refresh, a violating interval redone as two halves);
- **wall clock**: batch 16, inside ``lax.scan``.

Output: data/results/e10/w1spec_<tag>.json; model data/models/cfd/<name>.{npz,json}

Usage (guarded launcher; GPU, JAX env)
  scripts/safe_run.sh cfd_w1spec env -u LD_LIBRARY_PATH -u CUDA_HOME PYTHONPATH=src \\
      .venvs/frost-mhd/bin/python experiments/e10_cfd/w1spec.py --tag spec1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from functools import partial
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from frost.cfd import closure, schedule, spectral
from frost.cfd.spectral import Grid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_closure import DATA, adam_init, adam_update, load_model, save_model  # noqa: E402
from w1_eval import RES, best_of, correlation_times  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DT, REC = 0.007, 16


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--form", choices=("stencil", "tensor"), default="stencil")
    ap.add_argument("--bound", type=float, default=0.5, help="stencil form: |a| <= bound * u_scale")
    ap.add_argument("--stages", default="4:500,16:1000,32:2000")
    ap.add_argument("--train-k", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--init", type=Path, default=None)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--model", type=Path, default=ROOT / "data" / "models" / "cfd" / "kolmogorov64_spec_corr")
    ap.add_argument("--ks", default="4,8,16")
    ap.add_argument("--cert", default="", help="'k:e_star,...'")
    ap.add_argument("--batches", default="16")
    ap.add_argument("--steps", type=int, default=1024)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    g, g128 = Grid(64), Grid(128)
    with h5py.File(args.data, "r") as fh:
        train = np.asarray(fh["train/w64"])
        ev = np.asarray(fh["eval/w64"])
        ev128 = np.asarray(fh["eval/w128"])
    n_rec = ev.shape[0] - 1
    truth = jnp.asarray(ev[1:])
    to_hat = lambda a: jnp.asarray(np.fft.rfft2(a) * g.mask, jnp.complex64)  # noqa: E731
    u, v = spectral.velocity(g, to_hat(train[::50].reshape(-1, 64, 64)))
    scale = {"u": float(jnp.sqrt(jnp.mean(u**2 + v**2) / 2)), "w": float(np.sqrt(np.mean(train**2))),
             "k": 5e-3, "s": 0.1}
    if args.form == "stencil":
        scale["bound"] = args.bound
    print(f"devices {jax.devices()} | train {train.shape} | form {args.form}", flush=True)

    def tau_of(states):
        rho = np.asarray(spectral.correlation(spectral.to_real(g, states), truth))
        return correlation_times(np.where(np.isfinite(rho), rho, -1.0), REC * DT)

    # ---------------------------------------------------------------- training
    if args.skip_train:
        model = load_model(args.model)
        model["scale"].update({k: v for k, v in scale.items() if k not in model["scale"]})
    else:
        out_ch = closure.N_RAW[args.form]
        params = (load_model(args.init)["params"] if args.init is not None else
                  closure.init_cnn(jax.random.PRNGKey(0), hidden=args.hidden, layers=args.layers, out_ch=out_ch))
        data = jnp.asarray(train)
        data_hat = to_hat(train)
        ref_norm = float(np.mean(train**2))
        kk = args.train_k

        def loss_fn(p, w0, targets):
            model = {"params": p, "scale": scale}

            @jax.checkpoint
            def outer(wh, _):
                coef = closure.coefficients(g, model, wh)

                def inner(x, _):
                    x = spectral.step(g, x, DT, lambda y: closure.apply_operator(g, coef, y))
                    return x, spectral.to_real(g, x)

                return jax.lax.scan(inner, wh, None, length=kk)

            _, traj = jax.lax.scan(outer, w0, None, length=targets.shape[0] // kk)
            return jnp.mean((traj.reshape(targets.shape) - targets) ** 2) / ref_norm

        @partial(jax.jit, static_argnames=("unroll",))
        def train_step(p, opt, d, dh, it_t, it_b, lr, unroll: int):
            offs = it_t[None, :] + jnp.arange(1, unroll + 1)[:, None]
            loss, grads = jax.value_and_grad(loss_fn)(p, dh[it_t, it_b], d[offs, it_b[None, :]])
            finite = jnp.isfinite(loss) & jnp.all(jnp.asarray(
                [jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(grads)]))
            p_new, opt_new, _ = adam_update(p, grads, opt, lr)
            keep = lambda a, b: jnp.where(finite, a, b)  # noqa: E731
            return jax.tree_util.tree_map(keep, p_new, p), jax.tree_util.tree_map(keep, opt_new, opt), loss, finite

        def quick_tau(p):
            st = schedule.rollout(g, {"params": p, "scale": scale}, to_hat(ev[0, :4]), mode="operator",
                                  k=1, dt=DT, n_records=n_rec, record_every=REC)
            rho = np.asarray(spectral.correlation(spectral.to_real(g, st), truth[:, :4]))
            return float(correlation_times(np.where(np.isfinite(rho), rho, -1.0), REC * DT).mean())

        stages = [tuple(int(x) for x in s.split(":")) for s in args.stages.split(",")]
        total = sum(n for _, n in stages)
        opt, rng, it, skipped, t0 = adam_init(params), np.random.default_rng(0), 0, 0, time.perf_counter()
        print(f"start quick tau {quick_tau(params):.2f} | CNN {closure.cnn_flops(params, 64) / 1e9:.2f} GFLOP | "
              f"train hold k {kk}", flush=True)
        last_skipped = 0
        for unroll, n_iter in stages:
            for _ in range(n_iter):
                lr = args.lr * 0.5 * (1 + math.cos(math.pi * it / total)) + 1e-6
                if args.warmup:
                    lr *= min(1.0, (it + 1) / args.warmup)
                it_t = jnp.asarray(rng.integers(0, train.shape[0] - unroll - 1, args.batch))
                it_b = jnp.asarray(rng.integers(0, train.shape[1], args.batch))
                params, opt, loss, finite = train_step(params, opt, data, data_hat, it_t, it_b, lr, unroll)
                it += 1
                skipped += int(not bool(finite))
                if it % 250 == 0:
                    print(f"it {it:5d} unroll {unroll:3d} loss {float(loss):.4e} skipped {skipped} | "
                          f"{time.perf_counter() - t0:.0f} s", flush=True)
                    if skipped - last_skipped > 225:
                        raise SystemExit(f"training stuck: {skipped - last_skipped} of the last 250 non-finite")
                    last_skipped = skipped
            print(f"stage unroll {unroll} done: quick tau {quick_tau(params):.2f}", flush=True)
        model = {"params": params, "scale": scale}
        save_model(args.model, model, {"grid": 64, "dt": DT, "form": args.form, "stages": args.stages,
                                       "train_k": kk, "skipped_updates": skipped,
                                       "train_seconds": time.perf_counter() - t0,
                                       "gflop_per_call": closure.cnn_flops(params, 64) / 1e9})
        del data, data_hat

    # ---------------------------------------------------------------- scores
    w0 = to_hat(ev[0])
    rows: dict[str, dict] = {}

    def score(name, states, extra=None):
        tau = tau_of(states)
        rows[name] = {"tau": tau.tolist(), "tau_mean": float(tau.mean()),
                      "tau_sem": float(tau.std(ddof=1) / np.sqrt(len(tau))),
                      "finite": bool(jnp.all(jnp.isfinite(states.real))), **(extra or {})}
        print(f"{name:20s} tau {tau.mean():5.2f} +- {rows[name]['tau_sem']:.2f} | finite {rows[name]['finite']}"
              + (f" | {extra}" if extra else ""), flush=True)

    ks = [int(x) for x in args.ks.split(",") if x]
    cert = [(int(c.split(":")[0]), float(c.split(":")[1])) for c in args.cert.split(",") if c]
    score("none:1", schedule.rollout(g, None, w0, mode="none", k=1, dt=DT, n_records=n_rec, record_every=REC))
    score("smag:1", schedule.rollout(g, None, w0, mode="smag", k=1, dt=DT, n_records=n_rec, record_every=REC))
    for k in [1] + ks:
        score(f"operator:{k}", schedule.rollout(g, model, w0, mode="operator", k=k, dt=DT, n_records=n_rec,
                                                record_every=REC))
    for k_c, e_c in cert:
        st, n_rb, e_mean = schedule.cert_rollout(g, model, w0, k=k_c, dt=DT, e_star=e_c, n_records=n_rec,
                                                 record_every=REC)
        score(f"cert:{k_c}:{e_c}", st, {"rollbacks": int(n_rb), "intervals": n_rec * REC // k_c,
                                        "mean_hold_error": float(e_mean)})
    st128 = schedule.rollout(g128, None, jnp.asarray(np.fft.rfft2(ev128[0]) * g128.mask, jnp.complex64),
                             mode="none", k=1, dt=DT / 2, n_records=n_rec, record_every=2 * REC)
    score("spectral128:1", spectral.truncate(g128, g, st128))
    base = np.asarray(rows["operator:1"]["tau"])
    for r in rows.values():
        d = np.asarray(r["tau"]) - base
        r["dtau_vs_B0"], r["dtau_sem"] = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))

    # ---------------------------------------------------------------- wall clock
    timing = {}
    for b in (int(x) for x in args.batches.split(",")):
        wb = jnp.tile(w0, (-(-b // w0.shape[0]), 1, 1))[:b]
        wb128 = jnp.tile(jnp.asarray(np.fft.rfft2(ev128[0]) * g128.mask, jnp.complex64), (-(-b // 16), 1, 1))[:b]
        t = {}
        t["none:1"] = 1e3 * best_of(lambda: schedule.advance(g, None, wb, mode="none", k=1, dt=DT,
                                                             n_outer=args.steps))[0] / args.steps
        for k in [1] + ks:
            t[f"operator:{k}"] = 1e3 * best_of(lambda: schedule.advance(g, model, wb, mode="operator", k=k, dt=DT,
                                                                        n_outer=args.steps // k))[0] / args.steps
        for k_c, e_c in cert:
            t[f"cert:{k_c}:{e_c}"] = 1e3 * best_of(lambda: schedule.cert_advance(
                g, model, wb, k=k_c, dt=DT, e_star=e_c, n_outer=args.steps // k_c))[0] / args.steps
        t["spectral128:1"] = 1e3 * best_of(lambda: schedule.advance(g128, None, wb128, mode="none", k=1, dt=DT / 2,
                                                                    n_outer=2 * args.steps))[0] / args.steps
        timing[str(b)] = {"ms_per_step": t, "speedup_vs_B0": {n: t["operator:1"] / v for n, v in t.items()}}
        print(f"batch {b}: " + " | ".join(f"{n} {v:.3f}" for n, v in t.items()) + " ms", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    out = RES / f"w1spec_{args.tag}.json"
    out.write_text(json.dumps({"accuracy": rows, "timing": timing, "model": str(args.model), "form": args.form},
                              indent=1))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
