"""W1 on the finite-volume solver with learned interpolation: train, score, time (tests 11-12, FV).

1. **Reference.** The 1024^2 DNS of ``kolmogorov_dns.py``, filtered to the 64^2
   band. It is moved to cell centres by an exact Fourier half-cell phase shift: the
   spectral samples sit at i dx, finite-volume values at (i + 1/2) dx, and a
   half-cell mismatch would cap every correlation silently.
2. **Training** (Kochkov's recipe on our ``frost.cfd.finite_volume``): learned
   interpolation trained through the solver on unrolls of 4, 16 and 32 steps
   against the reference. Non-finite updates are skipped.
3. **Scores**, on 16 held-out trajectories over 15 time units: correlation time
   (rho < 0.95), paired against B0 = ``li:1``.
   - Baselines: upwind3 and central at 64^2, upwind3 at 128^2 (restricted by 2x2
     averaging), spectral DS at 128^2 (the strongest classical baseline here).
   - Holds: the coefficients (``li``, operator-consumed) and the learned flux
     correction (``li_value``), at k = 2, 4, 8, 16.
4. **Wall clock**: ms per 64^2-equivalent step inside ``lax.scan``, best of three
   windows, at batch 1 and 16.

Output: data/results/e10/w1fv_<tag>.json; model data/models/cfd/kolmogorov64_fv_li.{npz,json}

Usage (guarded launcher; GPU, JAX env)
  scripts/safe_run.sh cfd_w1fv env -u LD_LIBRARY_PATH -u CUDA_HOME PYTHONPATH=src \\
      .venvs/frost-mhd/bin/python experiments/e10_cfd/w1fv.py --tag fv1
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
from frost.cfd import finite_volume as fv
from frost.cfd.spectral import Grid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_closure import DATA, adam_init, adam_update, load_model, save_model  # noqa: E402
from w1_eval import RES, best_of, correlation_times  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "data" / "models" / "cfd" / "kolmogorov64_fv_li"
DT, REC = 0.007, 16


def half_cell(field: np.ndarray) -> np.ndarray:
    """Band-limited samples at i dx -> values at (i + 1/2) dx, exactly (Fourier phase)."""
    n = field.shape[-1]
    k = (np.fft.fftfreq(n) * n)[:, None]
    q = (np.fft.rfftfreq(n) * n)[None, :]
    h = 2 * np.pi / n
    return np.fft.irfft2(np.fft.rfft2(field) * np.exp(1j * (k + q) * h / 2), s=(n, n)).astype(np.float32)


def restrict(field, factor: int):
    *lead, n, _ = field.shape
    return field.reshape(*lead, n // factor, factor, n // factor, factor).mean(axis=(-3, -1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", default="4:500,16:1000,32:2000")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=0,
                    help="linear learning-rate warmup (iterations). An 8x96 network at lr 1e-3 "
                    "saturated its stencils within three Adam steps and never recovered")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--ks", default="2,4,8,16")
    ap.add_argument("--modes", default="li,li_value", help="hold modes scored at every k")
    ap.add_argument("--batches", default="1,16")
    ap.add_argument("--steps", type=int, default=1024)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--model", type=Path, default=MODEL)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--base", choices=("central", "upwind3"), default="upwind3")
    ap.add_argument("--velocity", choices=("avg", "cgrid"), default="avg",
                    help="face-velocity scheme of the finite-volume solver (all arms)")
    ap.add_argument("--stencil-width", type=int, default=3, help="learned stencil width across a face")
    ap.add_argument("--train-k", type=int, default=1,
                    help="schedule-aware training: refresh the learned stencil only every k steps "
                    "inside the unroll and hold it (operator) in between, so the network learns "
                    "coefficients that stay valid across the hold (R5 in CFD)")
    ap.add_argument("--init", type=Path, default=None, help="warm start from these weights")
    ap.add_argument("--distill-from", type=Path, default=None,
                    help="train this (small) model to reproduce that model's stencil coefficients "
                    "on the training states, as Frost's MD student is trained on teacher labels")
    ap.add_argument("--distill-iters", type=int, default=4000)
    ap.add_argument("--cert", default="",
                    help="certified intervals to score, 'k:e_star,...' (hold error checked at each "
                    "refresh; a violating interval is redone as two halves)")
    ap.add_argument("--student", type=Path, default=None,
                    help="a trained small model: enables li_student, li_res, li_res_adv")
    ap.add_argument("--bound", type=float, default=0.25)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    g, g128 = fv.FVGrid(64, velocity=args.velocity), fv.FVGrid(128, velocity=args.velocity)
    with h5py.File(args.data, "r") as fh:
        train = half_cell(np.asarray(fh["train/w64"]))
        ev = half_cell(np.asarray(fh["eval/w64"]))
        ev128 = half_cell(np.asarray(fh["eval/w128"]))
        ev_spec = np.asarray(fh["eval/w64"])  # grid-point samples, for the spectral baseline
        ev128_spec = np.asarray(fh["eval/w128"])
    print(f"devices {jax.devices()} | train {train.shape} | eval {ev.shape}", flush=True)
    ux, vy = fv.face_velocities(g, jnp.asarray(train[::50].reshape(-1, 64, 64)))
    scale = {"u": float(jnp.sqrt(jnp.mean(ux**2 + vy**2) / 2)), "w": float(np.sqrt(np.mean(train**2))),
             "bound": args.bound}
    static = closure.Static(base=args.base)
    truth = jnp.asarray(ev[1:])
    n_rec = ev.shape[0] - 1

    def tau_of(states):
        rho = np.asarray(spectral.correlation(states, truth))
        return correlation_times(np.where(np.isfinite(rho), rho, -1.0), REC * DT)

    # ---------------------------------------------------------------- training
    if args.distill_from is not None:
        teacher = load_model(args.distill_from)
        teacher["static"] = static
        params = closure.init_cnn(jax.random.PRNGKey(1), hidden=args.hidden, layers=args.layers,
                                  out_ch=fv.N_LI)
        print(f"student CNN {args.layers} x {args.hidden}: {closure.cnn_flops(params, 64) / 1e9:.3f} "
              f"GFLOP/call", flush=True)
        data = jnp.asarray(train)

        def dloss(p, w):
            c_t = jax.lax.stop_gradient(fv.li_coefficients(g, teacher, w))
            c_s = fv.li_coefficients(g, {"params": p, "scale": scale, "static": static}, w)
            return jnp.mean((c_s - c_t) ** 2) / jnp.mean(c_t**2)

        @jax.jit
        def dstep(p, opt, it_t, it_b, lr):
            loss, grads = jax.value_and_grad(dloss)(p, data[it_t, it_b])
            p, opt, _ = adam_update(p, grads, opt, lr)
            return p, opt, loss

        opt, rng, t0 = adam_init(params), np.random.default_rng(1), time.perf_counter()
        for it in range(args.distill_iters):
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * it / args.distill_iters)) + 1e-5
            params, opt, loss = dstep(params, opt, jnp.asarray(rng.integers(0, train.shape[0], args.batch)),
                                      jnp.asarray(rng.integers(0, train.shape[1], args.batch)), lr)
            if (it + 1) % 500 == 0:
                print(f"distill it {it + 1} relative coefficient MSE {float(loss):.4f} | "
                      f"{time.perf_counter() - t0:.0f} s", flush=True)
        model = {"params": params, "scale": scale, "static": static}
        save_model(args.model, {"params": params, "scale": scale},
                   {"grid": 64, "dt": DT, "form": "fv_li", "base": args.base, "distilled_from":
                    str(args.distill_from), "iters": args.distill_iters,
                    "gflop_per_call": closure.cnn_flops(params, 64) / 1e9})
        del data
    elif args.skip_train:
        model = load_model(args.model)
        meta = json.loads(args.model.with_suffix(".json").read_text())
        model["static"] = closure.Static(base=meta.get("base", "central"))
    else:
        params = (load_model(args.init)["params"] if args.init is not None else
                  closure.init_cnn(jax.random.PRNGKey(0), hidden=args.hidden, layers=args.layers,
                                   out_ch=fv.n_li(args.stencil_width)))
        print(f"CNN {args.layers} x {args.hidden}: {closure.cnn_flops(params, 64) / 1e9:.2f} GFLOP/call"
              f" | training hold k = {args.train_k}", flush=True)
        data = jnp.asarray(train)
        ref_norm = float(np.mean(train**2))

        def loss_fn(p, w0, targets):
            model = {"params": p, "scale": scale, "static": static}

            kk = args.train_k

            @jax.checkpoint
            def outer(w, _):
                coef = fv.li_coefficients(g, model, w)  # refreshed once, held for kk steps

                def inner(x, _):
                    x = fv.step(g, x, DT, "li", coef, args.base)
                    return x, x

                return jax.lax.scan(inner, w, None, length=kk)

            _, traj = jax.lax.scan(outer, w0, None, length=targets.shape[0] // kk)
            traj = traj.reshape(targets.shape)
            return jnp.mean((traj - targets) ** 2) / ref_norm

        @partial(jax.jit, static_argnames=("unroll",))
        def train_step(p, opt, d, it_t, it_b, lr, unroll: int):
            offs = it_t[None, :] + jnp.arange(1, unroll + 1)[:, None]
            loss, grads = jax.value_and_grad(loss_fn)(p, d[it_t, it_b], d[offs, it_b[None, :]])
            finite = jnp.isfinite(loss) & jnp.all(jnp.asarray(
                [jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(grads)]))
            p_new, opt_new, _ = adam_update(p, grads, opt, lr)
            keep = lambda a, b: jnp.where(finite, a, b)  # noqa: E731
            return jax.tree_util.tree_map(keep, p_new, p), jax.tree_util.tree_map(keep, opt_new, opt), loss, finite

        def quick_tau(p):
            st = fv.rollout(g, {"params": p, "scale": scale, "static": static}, jnp.asarray(ev[0, :4]),
                            mode="li", k=1,
                            dt=DT, n_records=n_rec, record_every=REC)
            rho = np.asarray(spectral.correlation(st, truth[:, :4]))
            return float(correlation_times(np.where(np.isfinite(rho), rho, -1.0), REC * DT).mean())

        stages = [tuple(int(x) for x in s.split(":")) for s in args.stages.split(",")]
        total = sum(n for _, n in stages)
        opt, rng, it, skipped, t0 = adam_init(params), np.random.default_rng(0), 0, 0, time.perf_counter()
        print(f"untrained (= {args.base}) quick tau {quick_tau(params):.2f}", flush=True)
        history = []
        for unroll, n_iter in stages:
            for _ in range(n_iter):
                lr = args.lr * 0.5 * (1 + math.cos(math.pi * it / total)) + 1e-5
                if args.warmup:
                    lr *= min(1.0, (it + 1) / args.warmup)
                it_t = jnp.asarray(rng.integers(0, train.shape[0] - unroll - 1, args.batch))
                it_b = jnp.asarray(rng.integers(0, train.shape[1], args.batch))
                params, opt, loss, finite = train_step(params, opt, data, it_t, it_b, lr, unroll)
                it += 1
                skipped += int(not bool(finite))
                if it % 250 == 0:
                    history.append((it, unroll, float(loss)))
                    print(f"it {it:5d} unroll {unroll:2d} loss {float(loss):.4e} skipped {skipped} | "
                          f"{time.perf_counter() - t0:.0f} s", flush=True)
                    window = skipped - (history[-2][3] if len(history) > 1 and len(history[-2]) > 3 else 0)
                    history[-1] = (*history[-1], skipped)
                    if window > 225:  # >90% of the last 250 updates were non-finite: stuck, stop
                        raise SystemExit(f"training stuck: {window} of the last 250 updates non-finite")
            print(f"stage unroll {unroll} done: quick tau (4 eval runs) {quick_tau(params):.2f}", flush=True)
        model = {"params": params, "scale": scale, "static": static}
        save_model(args.model, {"params": params, "scale": scale},
                   {"grid": 64, "dt": DT, "form": "fv_li", "stages": args.stages, "base": args.base,
                                       "skipped_updates": skipped, "history": history,
                                       "train_seconds": time.perf_counter() - t0,
                                       "gflop_per_call": closure.cnn_flops(params, 64) / 1e9})
        del data

    if args.student is not None:
        model["student"] = load_model(args.student)["params"]

    # ---------------------------------------------------------------- scores
    w0 = jnp.asarray(ev[0])
    rows: dict[str, dict] = {}

    def score(name, states):
        tau = tau_of(states)
        rows[name] = {"tau": tau.tolist(), "tau_mean": float(tau.mean()),
                      "tau_sem": float(tau.std(ddof=1) / np.sqrt(len(tau))),
                      "finite": bool(jnp.all(jnp.isfinite(states)))}
        print(f"{name:18s} tau {tau.mean():5.2f} +- {rows[name]['tau_sem']:.2f} | finite "
              f"{rows[name]['finite']}", flush=True)

    ks = [int(x) for x in args.ks.split(",")]
    configs = [("upwind3", 1), ("central", 1), ("li", 1)] + [(m, k) for m in args.modes.split(",")
                                                              for k in ks]
    if args.student is not None:
        configs.insert(3, ("li_student", 1))
    for mode, k in configs:
        score(f"{mode}:{k}", fv.rollout(g, model, w0, mode=mode, k=k, dt=DT, n_records=n_rec,
                                        record_every=REC))
    cert = [(int(c.split(":")[0]), float(c.split(":")[1])) for c in args.cert.split(",") if c]
    for k_c, e_c in cert:
        st, n_rb, e_mean = fv.cert_rollout(g, model, w0, k=k_c, dt=DT, e_star=e_c, n_records=n_rec,
                                           record_every=REC)
        score(f"cert:{k_c}:{e_c}", st)
        rows[f"cert:{k_c}:{e_c}"].update({"rollbacks": int(n_rb), "intervals": n_rec * REC // k_c,
                                          "mean_hold_error": float(e_mean)})
        print(f"  cert k {k_c} e* {e_c}: {int(n_rb)} of {n_rec * REC // k_c} intervals rolled back, "
              f"mean e {float(e_mean):.3f}", flush=True)
    st = fv.rollout(g128, None, jnp.asarray(ev128[0]), mode="upwind3", k=1, dt=DT / 2, n_records=n_rec,
                    record_every=2 * REC)
    score("fv128_upwind3:1", restrict(st, 2))
    gs = Grid(128)
    st = schedule.rollout(gs, None, jnp.asarray(np.fft.rfft2(ev128_spec[0]) * gs.mask, jnp.complex64),
                          mode="none", k=1, dt=DT / 2, n_records=n_rec, record_every=2 * REC)
    st = spectral.to_real(Grid(64), spectral.truncate(gs, Grid(64), st))
    score("spectral128:1", jnp.asarray(half_cell(np.asarray(st))))
    del ev_spec
    base = np.asarray(rows["li:1"]["tau"])
    for r in rows.values():
        d = np.asarray(r["tau"]) - base
        r["dtau_vs_B0"], r["dtau_sem"] = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
        r["matched_accuracy"] = bool(r["dtau_vs_B0"] >= -2 * r["dtau_sem"])

    # ---------------------------------------------------------------- wall clock
    timing = {}
    for b in (int(x) for x in args.batches.split(",")):
        rep = -(-b // w0.shape[0])
        wb = jnp.tile(w0, (rep, 1, 1))[:b]
        wb128 = jnp.tile(jnp.asarray(ev128[0]), (rep, 1, 1))[:b]
        ws128 = jnp.tile(jnp.asarray(np.fft.rfft2(ev128_spec[0]) * gs.mask, jnp.complex64), (rep, 1, 1))[:b]
        t = {}
        coef_fn = jax.jit(lambda w: fv.li_coefficients(g, model, w))
        t["cnn_call_ms"] = 1e3 * best_of(lambda: coef_fn(wb))[0]
        for mode, k in configs:
            t[f"{mode}:{k}"] = 1e3 * best_of(lambda: fv.advance(g, model, wb, mode=mode, k=k, dt=DT,
                                                                n_outer=args.steps // k))[0] / args.steps
        for k_c, e_c in cert:
            t[f"cert:{k_c}:{e_c}"] = 1e3 * best_of(lambda: fv.cert_advance(
                g, model, wb, k=k_c, dt=DT, e_star=e_c, n_outer=args.steps // k_c))[0] / args.steps
        t["fv128_upwind3:1"] = 1e3 * best_of(lambda: fv.advance(g128, None, wb128, mode="upwind3", k=1,
                                                                dt=DT / 2, n_outer=2 * args.steps))[0] / args.steps
        t["spectral128:1"] = 1e3 * best_of(lambda: schedule.advance(gs, None, ws128, mode="none", k=1, dt=DT / 2,
                                                                    n_outer=2 * args.steps))[0] / args.steps
        b0 = t["li:1"]
        timing[str(b)] = {"ms_per_step": t, "speedup_vs_B0": {n: b0 / v for n, v in t.items() if n != "cnn_call_ms"},
                          "closure_share_of_B0": (b0 - t["central:1"]) / b0}
        print(f"batch {b}: " + " | ".join(f"{n} {v:.3f}" for n, v in t.items()) + " ms", flush=True)
        print(f"batch {b}: closure share {timing[str(b)]['closure_share_of_B0']:.2f} | speedups " +
              " ".join(f"{n} {v:.2f}x" for n, v in timing[str(b)]["speedup_vs_B0"].items()), flush=True)

    RES.mkdir(parents=True, exist_ok=True)
    out = RES / f"w1fv_{args.tag}.json"
    out.write_text(json.dumps({"accuracy": rows, "timing": timing, "model": str(args.model)}, indent=1))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
