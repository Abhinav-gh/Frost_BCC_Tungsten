"""W1-T, learned: train the positive learned tracer scheme (``li_pp``) on 64^2 and score it in
the gate's frame (``tracer_frontier.py``), with Frost's holds.

1. **Training data.** The W1 *train* flows (record 0 of ``train/w128``), run by filtered
   spectral 64^2 at dt 0.014 as in the gate, each carrying T tracers drawn with keys the
   gate does not use. They are transported by spectral 512^2 (exp filter; it matches the
   gate's 1024^2 reference to 2e-3 relative L2 on the 64^2 band over 10 time units) and
   recorded every interval as exact 64^2 cell averages.
2. **Training** through the scheme (WENO-Z + learned stencil + flux-corrected positivity,
   SSP-RK3 at the gate's 64^2 substeps), coefficients refreshed every 0.014 interval:
   MSE on cell averages, normalised by the target variance, unrolls of 4, 16 and 32
   intervals. Non-finite updates are skipped.
3. **Scores** on the gate's eval flows and tracers, against its cached 1024^2 reference:
   tau on the |k| <= 21 and <= 31 bands, relative L2, minimum, mass; the coefficients held
   k = 1, 2, 4, 8 intervals (operator-consumed holds); wall clock at 64 fields.

Output: data/results/e10/tracer_learned_<tag>.json; model data/models/cfd/tracer64_li_<tag>.{npz,json}

Usage (guarded launcher; GPU, JAX env)
  scripts/safe_run.sh cfd_tracer_li env -u LD_LIBRARY_PATH -u CUDA_HOME PYTHONPATH=src \\
      .venvs/frost-mhd/bin/python experiments/e10_cfd/tracer_learned.py --tag li1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from frost.cfd import closure, spectral, tracers
from frost.cfd import finite_volume as fv
from frost.cfd.spectral import Grid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tracer_frontier import BANDS, DT_V, G64E, REC_I, ROOT, band, substeps  # noqa: E402
from train_closure import DATA, adam_init, adam_update, load_model, save_model  # noqa: E402
from w1_eval import RES, best_of, correlation_times  # noqa: E402

G512 = Grid(512, dealias="exp")
SCALE = {"c": 1.5, "u": 3.0, "bound": 0.5}


def flow_history(w128, n_int: int):
    w0 = spectral.truncate(Grid(128), G64E, jnp.asarray(np.fft.rfft2(w128) * Grid(128).mask, jnp.complex64))
    return jax.jit(lambda w: jax.lax.scan(lambda x, _: (spectral.step(G64E, x, DT_V), x), w, None,
                                          length=n_int)[1])(w0)


def cell_average_64(c_hat512):
    """Exact 64^2 cell averages of a 512^2 spectral field: the 64-cell sinc, then every 8th point."""
    h = 0.5 * G64E.dx
    f = (np.sinc(G512.kx * h / np.pi) * np.sinc(G512.ky * h / np.pi)).astype(np.float32)
    return spectral.to_real(G512, c_hat512 * f)[..., ::8, ::8]


@partial(jax.jit, static_argnames=("sub", "kappa"))
def reference_cells(c0_hat512, w_hist, *, sub: int, kappa: float):
    """Spectral 512^2 transport; 64^2 cell averages before every interval and after the last."""
    dt = DT_V / sub

    def interval(c, w64):
        u, v = tracers.point_velocity(G64E, G512, w64)
        c = jax.lax.fori_loop(0, sub, lambda _, x: tracers.spectral_step(G512, x, u[:, None], v[:, None], dt, kappa), c)
        return c, cell_average_64(c)

    _, cells = jax.lax.scan(interval, c0_hat512, w_hist)
    return jnp.concatenate([cell_average_64(c0_hat512)[None], cells])


def li_interval(model, c, ux, vy, coef, sub: int, kappa: float):
    dt = DT_V / sub
    return jax.lax.fori_loop(0, sub, lambda _, x: tracers.fv_step("li_pp", G64E.dx, x, ux, vy, dt, kappa, coef), c)


@partial(jax.jit, static_argnames=("k", "sub", "kappa", "n_rec"))
def rollout(model, c0, w_hist, *, k: int, sub: int, kappa: float, n_rec: int):
    """Coefficients refreshed every k intervals and held; records as the gate's (64^2 point-value spectra)."""
    ca = tracers.CellAverage(G64E)
    w_rec = w_hist[: n_rec * REC_I].reshape(n_rec, REC_I, *w_hist.shape[1:])

    def interval(carry, w64):
        c, coef, i = carry
        ux, vy = tracers.face_velocity(G64E, G64E, w64)
        ux, vy = ux[:, None], vy[:, None]
        coef = jax.lax.cond(i % k == 0, lambda: tracers.li_coefficients(model, c, ux, vy), lambda: coef)
        return (li_interval(model, c, ux, vy, coef, sub, kappa), coef, i + 1), None

    def record(carry, w_k):
        carry = jax.lax.scan(interval, carry, w_k)[0]
        c = carry[0]
        return carry, (tracers.truncate_sharp(G64E, 64, 31, ca.to_points(c)), c.min(axis=(-2, -1)),
                       c.mean(axis=(-2, -1)))

    coef0 = jnp.zeros((*c0.shape, fv.N_LI), c0.dtype)
    return jax.lax.scan(record, (c0, coef0, jnp.zeros((), jnp.int32)), w_rec)[1]


@partial(jax.jit, static_argnames=("k", "sub", "kappa"))
def advance(model, c0, w_hist, *, k: int, sub: int, kappa: float):
    def interval(carry, w64):
        c, coef, i = carry
        ux, vy = tracers.face_velocity(G64E, G64E, w64)
        ux, vy = ux[:, None], vy[:, None]
        coef = jax.lax.cond(i % k == 0, lambda: tracers.li_coefficients(model, c, ux, vy), lambda: coef)
        return (li_interval(model, c, ux, vy, coef, sub, kappa), coef, i + 1), None

    coef0 = jnp.zeros((*c0.shape, fv.N_LI), c0.dtype)
    return jax.lax.scan(interval, (c0, coef0, jnp.zeros((), jnp.int32)), w_hist)[0][0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", default="4:400,16:600,32:800", help="unroll (intervals):iterations,...")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tracers", type=int, default=4, help="training tracers per flow")
    ap.add_argument("--kappa", type=float, default=1e-4)
    ap.add_argument("--horizon", type=float, default=10.08)
    ap.add_argument("--holds", default="1,2,4,8")
    ap.add_argument("--eval-traj", type=int, default=8)
    ap.add_argument("--eval-tracers", type=int, default=2)
    ap.add_argument("--timed-intervals", type=int, default=64)
    ap.add_argument("--model", type=Path, default=None, help="score a saved model instead of training")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    n_int = round(args.horizon / DT_V)
    n_int -= n_int % REC_I
    with h5py.File(args.data, "r") as fh:
        w_train = np.asarray(fh["train/w128"][0])
        w_eval = np.asarray(fh["eval/w128"][0, : args.eval_traj])
    hist_eval = flow_history(w_eval, n_int)
    umax = float(jnp.max(jax.vmap(lambda w: spectral.max_speed(G64E, w))(hist_eval)))
    sub = substeps("li_pp", G64E, umax)
    print(f"devices {jax.devices()} | u_max {umax:.2f} | 64^2 substeps {sub} | kappa {args.kappa:g}", flush=True)

    out_model = ROOT / "data" / "models" / "cfd" / f"tracer64_li_{args.tag}"
    if args.model is None:
        # ---------------------------------------------------------- training data
        hist = flow_history(w_train, n_int)
        n_flow, n_tr = hist.shape[1], args.tracers
        c0 = tracers.random_tracer(G64E, jax.random.PRNGKey(11), batch=(n_flow, n_tr))
        sub512 = substeps("spectral", G512, umax)
        t0 = time.perf_counter()
        cells = reference_cells(spectral.pad(G64E, G512, c0), hist, sub=sub512, kappa=args.kappa)
        cells = cells.reshape(cells.shape[0], n_flow * n_tr, 64, 64)
        flow_of = jnp.asarray(np.repeat(np.arange(n_flow), n_tr))
        var = float(jnp.var(cells))
        print(f"training data: {n_flow} flows x {n_tr} tracers x {n_int} intervals at 512^2 "
              f"({time.perf_counter() - t0:.0f} s) | target variance {var:.3f}", flush=True)

        # ---------------------------------------------------------- training
        params = closure.init_cnn(jax.random.PRNGKey(5), in_ch=3, hidden=args.hidden, layers=args.layers,
                                  out_ch=fv.N_LI)

        def loss_fn(p, c_start, w_seq, targets):
            model = {"params": p, "scale": SCALE}

            def body(c, xs):
                w64, tgt = xs
                ux, vy = tracers.face_velocity(G64E, G64E, w64)
                coef = tracers.li_coefficients(model, c, ux, vy)
                c = li_interval(model, c, ux, vy, coef, sub, args.kappa)
                return c, jnp.mean((c - tgt) ** 2)

            return jnp.mean(jax.lax.scan(body, c_start, (w_seq, targets))[1]) / var

        @partial(jax.jit, static_argnames=("unroll",))
        def train_step(p, opt, hist, cells, flow_of, it_t, it_s, lr, unroll: int):
            offs = it_t[None, :] + jnp.arange(unroll)[:, None]
            w_seq = hist[offs, flow_of[it_s][None, :]]
            loss, grads = jax.value_and_grad(loss_fn)(p, cells[it_t, it_s], w_seq, cells[offs + 1, it_s[None, :]])
            finite = jnp.isfinite(loss) & jnp.all(jnp.asarray(
                [jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads)]))
            p_new, opt_new, _ = adam_update(p, grads, opt, lr)
            keep = lambda a, b: jnp.where(finite, a, b)  # noqa: E731
            return jax.tree_util.tree_map(keep, p_new, p), jax.tree_util.tree_map(keep, opt_new, opt), loss, finite

        opt, rng, it, skipped, t0 = adam_init(params), np.random.default_rng(0), 0, 0, time.perf_counter()
        stages = [tuple(int(x) for x in s.split(":")) for s in args.stages.split(",")]
        history = []
        for si, (unroll, n_iter) in enumerate(stages):
            lr = args.lr if si < len(stages) - 1 else args.lr / 3
            for _ in range(n_iter):
                it_t = jnp.asarray(rng.integers(0, n_int - unroll, args.batch))
                it_s = jnp.asarray(rng.integers(0, cells.shape[1], args.batch))
                params, opt, loss, finite = train_step(params, opt, hist, cells, flow_of, it_t, it_s, lr, unroll)
                it += 1
                skipped += int(not bool(finite))
                if it % 100 == 0:
                    history.append((it, unroll, float(loss)))
                    print(f"it {it:5d} unroll {unroll:2d} loss {float(loss):.4e} skipped {skipped} | "
                          f"{time.perf_counter() - t0:.0f} s", flush=True)
        model = {"params": params, "scale": SCALE}
        save_model(out_model, model, {"hidden": args.hidden, "kappa": args.kappa, "stages": args.stages,
                                      "skipped": skipped, "history": history})
        del cells
    else:
        model = load_model(args.model)
        model["scale"] = {**SCALE, **model["scale"]}

    # -------------------------------------------------------------- scores in the gate's frame
    cache = ROOT / "data" / "cfd" / f"tracer_ref1024_k{args.kappa:g}_T{n_int}_{args.eval_traj}x{args.eval_tracers}.npz"
    ref_spec = jnp.asarray(np.load(cache)["spec"])
    ref = {b: band(b, ref_spec) for b in BANDS}
    t_rec = REC_I * DT_V
    c0 = tracers.random_tracer(G64E, jax.random.PRNGKey(7), batch=(args.eval_traj, args.eval_tracers))
    x0 = tracers.CellAverage(G64E).from_points(c0)
    reps = -(-64 // (args.eval_traj * args.eval_tracers))
    xb, w_time = jnp.tile(x0, (1, reps, 1, 1)), hist_eval[: args.timed_intervals]
    rows = {}
    for k in (int(x) for x in args.holds.split(",")):
        spec, cmin, mean = rollout(model, x0, hist_eval, k=k, sub=sub, kappa=args.kappa, n_rec=n_int // REC_I)
        row = {"k": k, "min": float(jnp.min(cmin)), "mass_drift": float(jnp.max(jnp.abs(mean - 1.0)))}
        for b in BANDS:
            a = band(b, spec)
            rho = np.asarray(spectral.correlation(a, ref[b])).reshape(a.shape[0], -1)
            tau = correlation_times(np.where(np.isfinite(rho), rho, -1.0), t_rec)
            err = np.asarray(jnp.sqrt(((a - ref[b]) ** 2).sum((-2, -1)) / ((ref[b] - ref[b].mean((-2, -1), keepdims=True)) ** 2).sum((-2, -1))))
            err = err.reshape(err.shape[0], -1).mean(-1)
            row[f"tau_{b}"], row[f"tau_{b}_se"] = float(tau.mean()), float(tau.std(ddof=1) / np.sqrt(tau.size))
            row[f"relL2_{b}"] = {f"{t:g}": float(err[round(t / t_rec) - 1]) for t in (1.008, 2.016, 5.04, 10.08)}
        lo, hi = best_of(lambda k=k: advance(model, xb, w_time, k=k, sub=sub, kappa=args.kappa))
        row["ms_per_0.014"], row["window_ratio"] = 1e3 * lo / args.timed_intervals, lo / hi
        rows[f"li_pp64:k{k}"] = row
        print(f"li_pp64 k {k:2d} | tau k21 {row['tau_k21']:5.2f} +- {row['tau_k21_se']:.2f} | tau k31 "
              f"{row['tau_k31']:5.2f} | relL2 k31 {row['relL2_k31']} | min {row['min']:+.2e} | mass "
              f"{row['mass_drift']:.1e} | {row['ms_per_0.014']:.3f} ms per 0.014 (64 fields)", flush=True)

    out = RES / f"tracer_learned_{args.tag}.json"
    out.write_text(json.dumps({"model": str(out_model if args.model is None else args.model), "rows": rows,
                               "flops_per_call_per_field": closure.cnn_flops(model["params"], 64)}, indent=1))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
