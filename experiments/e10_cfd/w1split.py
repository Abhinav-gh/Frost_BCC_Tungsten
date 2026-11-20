"""W1, accuracy-first: the ML split operator (Dresdner et al. 2023) on the filtered spectral
solver; trained, scored against the classical frontier, and run under Frost's holds.

Why this model. Two findings of 2026-10-02 (new_directions.md S5):
- spectral DS with the exponential filter is a far stronger baseline than with the 2/3
  rule (64^2: tau 5.35 against 2.48; 96^2: 7.58), so a learned solver has to beat that;
- Dresdner et al. reach the accuracy of a ~2x finer spectral grid with this model, but
  report its network "about 10x slower than Physics-Step", which removes any speedup.
  Their suggested remedy, "applying corrections less frequently", is Frost's certified
  hold. This script measures whether the hold recovers the speedup at matched accuracy.

Pieces (``frost.cfd.split``): velocity-only inputs, an encoder-process-decoder CNN with
dilated convolutions, the correction as explicit Euler outside the RK4 stages, the
exponential filter on the physics step.

Data. Initial states come from the 128^2 DNS fields (``train/w128``, every 8th 64^2 step,
so they carry the 64^2 grid's full band); targets are the 64^2 fields (``train/w64``)
and the loss is the velocity error on the 64^2 2/3 band, the band every method is
scored on. ``w128[j]`` is the state at ``w64[8 j + 7]`` (kolmogorov_dns.py writes w128
after every 8th step).

Scores: tau over the 16 eval trajectories (scored on the 64^2 2/3 band) for the filtered
spectral solver at 64, 96, 112 and 128 (each at its largest stable step), the model every
step (B0), value and advected holds, certified intervals; paired delta-tau against B0.
Wall clock at batch 16 (latency-bound) and 256 (compute-bound), per 0.007 time units,
with both timing windows, all in one process so the GPU state is shared.

Output: data/results/e10/w1split_<tag>.json; model data/models/cfd/<name>.{npz,json}

Usage (guarded launcher; GPU, JAX env)
  scripts/safe_run.sh cfd_split env -u LD_LIBRARY_PATH -u CUDA_HOME PYTHONPATH=src \\
      .venvs/frost-mhd/bin/python experiments/e10_cfd/w1split.py --tag s1
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

from frost.cfd import schedule, spectral, split
from frost.cfd.spectral import Grid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_closure import DATA, adam_init, adam_update  # noqa: E402
from w1_eval import RES, best_of, correlation_times  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DT, REC = 0.007, 16
W128_EVERY, W128_OFFSET = 8, 7  # w128[j] is w64[8 j + 7]
CLASSICAL = ((96, 0.014), (112, 0.007), (128, 0.007))  # filtered spectral, largest stable step


def save_split(path: Path, model: dict, meta: dict) -> None:
    leaves = jax.tree_util.tree_leaves(model["params"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path.with_suffix(".npz"), *[np.asarray(x) for x in leaves])
    path.with_suffix(".json").write_text(json.dumps(
        {**meta, "scale_in": model["scale_in"], "scale_out": model["scale_out"],
         "rates": list(model["static"].rates)}, indent=1))


def load_split(path: Path) -> dict:
    meta = json.loads(path.with_suffix(".json").read_text())
    model = split.init_model(jax.random.PRNGKey(0), scale_in=meta["scale_in"], scale_out=meta["scale_out"],
                             rates=tuple(meta["rates"]), hidden=meta["hidden"], blocks=meta["blocks"])
    z = np.load(path.with_suffix(".npz"))
    treedef = jax.tree_util.tree_structure(model["params"])
    model["params"] = jax.tree_util.tree_unflatten(treedef, [jnp.asarray(z[f"arr_{i}"])
                                                             for i in range(treedef.num_leaves)])
    return model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", default="8:1000,16:2000,32:12000", help="unroll:iterations,...")
    ap.add_argument("--train-k", type=int, default=1, help="hold the correction k steps inside the unroll")
    ap.add_argument("--dt-mult", type=int, default=2, help="time step = 0.007 x this")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--ema", type=float, default=0.99)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--rates", default="1,2,4,2,1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", type=Path, default=None)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--model", type=Path, default=ROOT / "data" / "models" / "cfd" / "kolmogorov64_split")
    ap.add_argument("--ks", default="2,4,8,16", help="value holds to score")
    ap.add_argument("--advected", default="", help="advected holds to score")
    ap.add_argument("--cert", default="", help="'k:e_star,...'")
    ap.add_argument("--batches", default="16,256")
    ap.add_argument("--steps", type=int, default=512, help="timed length, in steps of 0.007")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--extra-data", default="", help="comma-separated h5 files with more train/ groups "
                    "(kolmogorov_dns.py --train-only), joined along the trajectory axis")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    dt = DT * args.dt_mult
    g, g23, g128 = Grid(64, dealias="exp"), Grid(64), Grid(128)
    with h5py.File(args.data, "r") as fh:
        train64 = np.asarray(fh["train/w64"])
        train128 = np.asarray(fh["train/w128"])
        ev = np.asarray(fh["eval/w64"])
        ev128 = np.asarray(fh["eval/w128"])
    for extra in (Path(x) for x in args.extra_data.split(",") if x):
        with h5py.File(extra, "r") as fh:
            train64 = np.concatenate([train64, np.asarray(fh["train/w64"])], axis=1)
            train128 = np.concatenate([train128, np.asarray(fh["train/w128"])], axis=1)
    n_rec = ev.shape[0] - 1
    truth = jnp.asarray(ev[1:])
    rec_steps = REC // args.dt_mult  # model steps per record
    hat128 = lambda a: jnp.asarray(np.fft.rfft2(a) * g128.mask, jnp.complex64)  # noqa: E731
    w0_eval = spectral.truncate(g128, g, hat128(ev128[0]))
    print(f"devices {jax.devices()} | train {train64.shape} + {train128.shape} | dt {dt} | "
          f"train hold k {args.train_k}", flush=True)

    def tau_of(states):
        rho = np.asarray(spectral.correlation(spectral.to_real(g23, states * g23.mask), truth[:, : states.shape[1]]))
        return correlation_times(np.where(np.isfinite(rho), rho, -1.0), REC * DT)

    # ---------------------------------------------------------------- training
    rates = tuple(int(r) for r in args.rates.split(","))
    meta = {"grid": 64, "dealias": "exp", "dt": dt, "hidden": args.hidden, "blocks": args.blocks}
    if args.skip_train:
        model = load_split(args.model)
    else:
        model = (load_split(args.init) if args.init is not None else
                 split.init_model(jax.random.PRNGKey(args.seed), rates=rates, hidden=args.hidden,
                                  blocks=args.blocks))
        ic = jnp.stack([spectral.truncate(g128, g, hat128(train128[j])) for j in range(train128.shape[0])])
        tg = jnp.asarray(np.fft.rfft2(train64) * g23.mask, jnp.complex64)
        ut, vt = spectral.velocity(g23, tg[::50])
        ref_norm = float(jnp.mean(ut**2 + vt**2))
        del train128
        kk, m = args.train_k, args.dt_mult
        template = {k: v for k, v in model.items() if k != "params"}

        def loss_fn(p, w0, targets):
            mdl = {**template, "params": p}

            @jax.checkpoint
            def outer(wh, _):
                c = split.correction(g, mdl, wh)
                return jax.lax.scan(lambda x, _: (split.split_step(g, x, c, dt),) * 2, wh, None, length=kk)

            _, traj = jax.lax.scan(outer, w0, None, length=targets.shape[0] // kk)
            traj = traj.reshape(targets.shape) * g23.mask
            up, vp = spectral.velocity(g23, traj)
            uq, vq = spectral.velocity(g23, targets)
            return jnp.mean((up - uq) ** 2 + (vp - vq) ** 2) / ref_norm

        def windows(ic, tg, j_idx, b_idx, unroll):
            t0 = W128_EVERY * j_idx + W128_OFFSET
            offs = t0[None, :] + m * jnp.arange(1, unroll + 1)[:, None]
            return ic[j_idx, b_idx], tg[offs, b_idx[None, :]]

        @partial(jax.jit, static_argnames=("unroll",))
        def eval_loss(p, ic, tg, j_idx, b_idx, unroll: int):
            return loss_fn(p, *windows(ic, tg, j_idx, b_idx, unroll))

        @partial(jax.jit, static_argnames=("unroll",))
        def train_step(p, opt, ema, ic, tg, j_idx, b_idx, lr, norm, unroll: int):
            # the loss is divided by the physics step's own error on these windows' unroll
            # length (1.0 = no better than the base): the raw error is ~1e-6 of the flow, and
            # at that scale Adam's eps swamps the gradient (the first run did not learn)
            loss, grads = jax.value_and_grad(lambda q: loss_fn(q, *windows(ic, tg, j_idx, b_idx, unroll)) / norm)(p)
            finite = jnp.isfinite(loss) & jnp.all(jnp.asarray(
                [jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(grads)]))
            p_new, opt_new, _ = adam_update(p, grads, opt, lr, b2=0.99)
            keep = lambda a, b: jnp.where(finite, a, b)  # noqa: E731
            p = jax.tree_util.tree_map(keep, p_new, p)
            opt = jax.tree_util.tree_map(keep, opt_new, opt)
            ema = jax.tree_util.tree_map(lambda e, q: args.ema * e + (1 - args.ema) * q, ema, p)
            return p, opt, ema, loss, finite

        def quick_tau(p):
            st = split.rollout(g, {**template, "params": p}, w0_eval[:4], mode="value", k=1, dt=dt,
                               n_records=n_rec, record_every=rec_steps)
            return float(tau_of(st).mean())

        stages = [tuple(int(x) for x in s.split(":")) for s in args.stages.split(",")]
        total = sum(n for _, n in stages)
        params = model["params"]
        opt, ema = adam_init(params), params
        rng, it, skipped, last_skipped, t_start = np.random.default_rng(args.seed), 0, 0, 0, time.perf_counter()
        print(f"start quick tau {quick_tau(params):.2f} | EPD {split.epd_flops(params, 64) / 1e9:.2f} GFLOP "
              f"per call | rates {rates}", flush=True)
        zero_p = jax.tree_util.tree_map(jnp.zeros_like, params)
        for unroll, n_iter in stages:
            j_max = (train64.shape[0] - 1 - W128_OFFSET - m * unroll) // W128_EVERY
            fixed = np.random.default_rng(123)
            jf = jnp.asarray(fixed.integers(0, j_max + 1, 64))
            bf = jnp.asarray(fixed.integers(0, train64.shape[1], 64))
            norm = float(eval_loss(zero_p, ic, tg, jf, bf, unroll))
            print(f"stage unroll {unroll}: physics-step loss {norm:.4e} (the unit); current model "
                  f"{float(eval_loss(params, ic, tg, jf, bf, unroll)) / norm:.4f}", flush=True)
            for _ in range(n_iter):
                lr = args.lr * 0.5 * (1 + math.cos(math.pi * it / total)) + 1e-6
                lr *= min(1.0, (it + 1) / args.warmup) if args.warmup else 1.0
                j_idx = jnp.asarray(rng.integers(0, j_max + 1, args.batch))
                b_idx = jnp.asarray(rng.integers(0, train64.shape[1], args.batch))
                params, opt, ema, loss, finite = train_step(params, opt, ema, ic, tg, j_idx, b_idx, lr, norm,
                                                            unroll)
                it += 1
                skipped += int(not bool(finite))
                if it % 250 == 0:
                    print(f"it {it:6d} unroll {unroll:3d} loss {float(loss):.4e} skipped {skipped} | "
                          f"{time.perf_counter() - t_start:.0f} s", flush=True)
                    if skipped - last_skipped > 225:
                        raise SystemExit(f"training stuck: {skipped - last_skipped} of the last 250 non-finite")
                    last_skipped = skipped
                if it % 2000 == 0:
                    print(f"  quick tau (4 eval runs) {quick_tau(ema):.2f} (EMA)", flush=True)
            print(f"stage unroll {unroll} done: quick tau {quick_tau(ema):.2f} (EMA) | fixed windows "
                  f"{float(eval_loss(ema, ic, tg, jf, bf, unroll)) / norm:.4f} of the physics step", flush=True)
        model = {**template, "params": ema}
        save_split(args.model, model, {**meta, "stages": args.stages, "train_k": kk, "skipped": skipped,
                                       "train_seconds": time.perf_counter() - t_start, "lr": args.lr})
        del ic, tg

    # ---------------------------------------------------------------- scores
    rows: dict[str, dict] = {}

    def score(name, states, extra=None):
        tau = tau_of(states)
        rows[name] = {"tau": tau.tolist(), "tau_mean": float(tau.mean()),
                      "tau_sem": float(tau.std(ddof=1) / np.sqrt(len(tau))),
                      "finite": bool(jnp.all(jnp.isfinite(states.real))), **(extra or {})}
        print(f"{name:24s} tau {tau.mean():5.2f} +- {rows[name]['tau_sem']:.2f} | finite {rows[name]['finite']}"
              + (f" | {extra}" if extra else ""), flush=True)

    ks = [int(x) for x in args.ks.split(",") if x]
    adv = [int(x) for x in args.advected.split(",") if x]
    cert = [(int(c.split(":")[0]), float(c.split(":")[1])) for c in args.cert.split(",") if c]
    score("spectral64exp", schedule.rollout(g, None, w0_eval, mode="none", k=1, dt=dt, n_records=n_rec,
                                            record_every=rec_steps))
    for k in [1] + ks:
        score(f"split:value:{k}", split.rollout(g, model, w0_eval, mode="value", k=k, dt=dt, n_records=n_rec,
                                                record_every=rec_steps))
    for k in adv:
        score(f"split:advected:{k}", split.rollout(g, model, w0_eval, mode="advected", k=k, dt=dt,
                                                   n_records=n_rec, record_every=rec_steps))
    for k_c, e_c in cert:
        st, n_rb, e_mean = split.cert_rollout(g, model, w0_eval, k=k_c, dt=dt, e_star=e_c, n_records=n_rec,
                                              record_every=rec_steps)
        score(f"split:cert:{k_c}:{e_c}", st, {"rollbacks": int(n_rb), "intervals": n_rec * rec_steps // k_c,
                                              "mean_hold_error": float(e_mean)})
    w0_classical = {}
    for n, cdt in CLASSICAL:
        gc = Grid(n, dealias="exp")
        w0_classical[n] = spectral.truncate(g128, gc, hat128(ev128[0]))
        st = schedule.rollout(gc, None, w0_classical[n], mode="none", k=1, dt=cdt, n_records=n_rec,
                              record_every=round(REC * DT / cdt))
        score(f"spectral{n}exp", spectral.truncate(gc, g23, st))
    base = np.asarray(rows["split:value:1"]["tau"])
    for r in rows.values():
        d = np.asarray(r["tau"]) - base
        r["dtau_vs_B0"], r["dtau_sem"] = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))

    # ---------------------------------------------------------------- wall clock
    timing = {}
    for b in (int(x) for x in args.batches.split(",")):
        tile = lambda w: jnp.tile(w, (-(-b // w.shape[0]), 1, 1))[:b]  # noqa: E731
        wb = tile(w0_eval)
        per = args.steps // args.dt_mult  # model steps in the timed window
        t, windows = {}, {}

        def timed(name, fn):
            lo, hi = best_of(fn)
            t[name], windows[name] = 1e3 * lo / args.steps, lo / hi

        timed("spectral64exp", lambda: schedule.advance(g, None, wb, mode="none", k=1, dt=dt, n_outer=per))
        net = jax.jit(lambda w: split.correction(g, model, w))
        timed("network_call", lambda: net(wb))
        t["network_call"] *= args.steps  # one call, not per 0.007
        for k in [1] + ks:
            timed(f"split:value:{k}", lambda: split.advance(g, model, wb, mode="value", k=k, dt=dt, n_steps=per))
        for k in adv:
            timed(f"split:advected:{k}", lambda: split.advance(g, model, wb, mode="advected", k=k, dt=dt,
                                                               n_steps=per))
        for k_c, e_c in cert:
            timed(f"split:cert:{k_c}:{e_c}", lambda: split.cert_advance(g, model, wb, k=k_c, dt=dt, e_star=e_c,
                                                                        n_steps=per))
        for n, cdt in CLASSICAL:
            gc, wc = Grid(n, dealias="exp"), tile(w0_classical[n])
            nc = round(args.steps * DT / cdt)
            timed(f"spectral{n}exp", lambda: schedule.advance(gc, None, wc, mode="none", k=1, dt=cdt, n_outer=nc))
        timing[str(b)] = {"ms_per_0.007": t, "window_ratio": windows,
                          "speedup_vs_B0": {n: t["split:value:1"] / v for n, v in t.items() if n != "network_call"}}
        print(f"batch {b}: " + " | ".join(f"{n} {v:.3f}" for n, v in t.items()) + " ms per 0.007 "
              f"(network_call: ms per call)", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    out = RES / f"w1split_{args.tag}.json"
    out.write_text(json.dumps({"accuracy": rows, "timing": timing, "model": str(args.model), "dt": dt},
                              indent=1))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
