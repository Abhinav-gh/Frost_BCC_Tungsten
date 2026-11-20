"""Workload W2: tile-sparse refresh of a trained MHD corrector on a blast wave (test 15).

Steps:
1. **Truth.** astronomix runs the MHD blast wave at 4x the coarse resolution, and the
   result is block-averaged to the coarse grid at every coarse step. Three training
   variants move the centre and the inner pressure; the evaluation case is the
   canonical blast (centre, p_in = 10).
2. **Corrector.** astronomix's ``CorrectorCNN`` design (three 3x3 layers, B
   through curl E; ``frost.cfd.mhd``), trained *through the solver*, as Storcks &
   Buck do: unrolls of 1, 4, 8 and 16 steps against the truth, in astronomix's
   public BACKWARDS differentiation mode. Non-finite updates are skipped. Density
   and pressure are corrected multiplicatively, so they stay positive.
   History (2026-10-01):
   - an a-priori corrector (73% of the correction's variance) blew up in every
     closed-loop run;
   - additive training through the solver NaN'd at 16-step unrolls.
3. **Schedules** on the evaluation case, all against the same truth:
   - no corrector;
   - B0 (every step);
   - M1 (dense refresh every k = 2, 4, 8);
   - M4 (budget m = n_tiles / k tiles per step, tiles of t = 16 and 32 cells,
     age_max = 2k).

   Reported per schedule: the relative L2 error of the final state against the
   truth and against B0, and the ms per step inside ``lax.scan``.

M1 and M4 at the same k do the same average CNN work, apart from M4's halo. The
comparison asks where a fixed refresh budget should go on a flow whose activity
sits on a front.

Output: data/results/e10/w2_blast_<tag>.json, model data/models/cfd/blast<n>_corrector.npz

Usage (guarded launcher; GPU, JAX env)
  scripts/safe_run.sh cfd_w2 env -u LD_LIBRARY_PATH -u CUDA_HOME PYTHONPATH=src \\
      .venvs/frost-mhd/bin/python experiments/e10_cfd/w2_blast.py --tag test15
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from frost.cfd import closure, mhd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_closure import adam_init, adam_update  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TRAIN_CASES = [((0.1, 0.0), 8.0), ((-0.1, 0.1), 12.0), ((0.05, -0.12), 10.0)]
EVAL_CASE = ((0.0, 0.0), 10.0)


def truth(n: int, factor: int, dt: float, steps: int, center, p_in) -> np.ndarray:
    fine = mhd.Solver(mhd.blast_wave(n * factor, center, p_in), n * factor, dt / factor)

    @jax.jit
    def coarse_step(s):
        s = jax.lax.scan(lambda c, _: (fine.step(c), None), s, None, length=factor)[0]
        return s, mhd.downsample(s, factor)

    out = [np.asarray(mhd.downsample(fine.state0, factor))]
    s = fine.state0
    for _ in range(steps):
        s, d = coarse_step(s)
        out.append(np.asarray(d))
    traj = np.stack(out)
    if not np.all(np.isfinite(traj)):
        raise SystemExit(f"fine run not finite for {center}, {p_in}")
    return traj  # (steps + 1, 7, n, n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--t-end", type=float, default=0.15)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--stages", default="1:400,4:400,8:400,16:400", help="unroll:iterations")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--ks", default="2,4,8")
    ap.add_argument("--tile-sizes", default="16,32")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cache", type=Path, default=None)
    args = ap.parse_args()
    n, f = args.n, args.factor
    cache = args.cache or ROOT / "data" / "cfd" / f"blast{n}_x{f}.npz"
    dt = min(mhd.stable_dt(mhd.blast_wave(n, c, p), n) for c, p in TRAIN_CASES + [EVAL_CASE])
    steps = int(args.t_end / dt)
    print(f"devices {jax.devices()} | coarse {n}^2, fine {n * f}^2 | dt {dt:.3e} | {steps} steps",
          flush=True)

    # ---- 1. truth
    if cache.exists():
        z = np.load(cache)
        train_trajs, ev = [z[f"train{i}"] for i in range(len(TRAIN_CASES))], z["eval"]
    else:
        t0 = time.perf_counter()
        train_trajs = [truth(n, f, dt, steps, c, p) for c, p in TRAIN_CASES]
        ev = truth(n, f, dt, steps, *EVAL_CASE)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, eval=ev, **{f"train{i}": t for i, t in enumerate(train_trajs)})
        print(f"truth: {len(TRAIN_CASES) + 1} fine runs in {time.perf_counter() - t0:.0f} s",
              flush=True)

    # ---- 2. corrector, trained through the solver
    coarse = mhd.Solver(mhd.blast_wave(n, *TRAIN_CASES[0]), n, dt, differentiable=True)
    step_b = jax.jit(jax.vmap(coarse.step))
    trajs = np.stack(train_trajs)  # (cases, steps + 1, 7, n, n)
    xs = trajs[:, :-1].reshape(-1, *trajs.shape[2:])
    nxt = np.concatenate([np.asarray(step_b(jnp.asarray(xs[i:i + 64]))) for i in range(0, len(xs), 64)])
    after = trajs[:, 1:].reshape(-1, *trajs.shape[2:])
    targets = (after - nxt) / dt
    for c in mhd.POSITIVE:  # relative rates for the multiplicative channels
        targets[:, c] = np.log(np.maximum(after[:, c], 1e-6) / np.maximum(nxt[:, c], 1e-6)) / dt
    # Bz is identically zero in this 2D in-plane problem. A zero-variance channel must
    # not set a scale: normalising by ~1e-6 amplified any stray Bz a millionfold, and
    # E_z (the one E component that corrects Bx, By) took Bz's ~1e-8 target scale.
    # Both NaN'd tests 15, 15b and 15c (2026-10-02).
    mean, std = xs.mean(axis=(0, 2, 3)), xs.std(axis=(0, 2, 3))
    std = np.maximum(std, 1e-3 * std.max())
    tstd = targets.std(axis=(0, 2, 3))
    tstd = np.maximum(tstd, 1e-3 * tstd.max())
    out_scale = tstd.copy()
    out_scale[4] = out_scale[5] = 0.0  # E_x, E_y: unused in-plane
    out_scale[6] = 0.5 * (tstd[4] + tstd[5]) / n  # E_z ~ (Bx, By correction) x dx
    scale = {"mean": mean.tolist(), "std": std.tolist(), "out": out_scale.tolist()}
    static = closure.Static(inplane=True)
    del xs, nxt, targets, after
    params = mhd.init_corrector(jax.random.PRNGKey(0), hidden=args.hidden)
    snorm = jnp.asarray(std[:, None, None])
    traj_d = jnp.asarray(trajs)

    def loss_fn(p, s0, truth_seq):
        model = {"params": p, "scale": scale, "static": static}

        @jax.checkpoint
        def body(st, _):
            st = jax.vmap(coarse.step)(st)
            st = mhd.apply_correction(st, mhd.correction(model, mhd.raw_output(model, st), 1.0 / n), dt)
            return st, st

        _, seq = jax.lax.scan(body, s0, None, length=truth_seq.shape[0])
        return jnp.mean(((seq - truth_seq) / snorm) ** 2)

    @partial(jax.jit, static_argnames=("unroll",))
    def train_step(p, opt, case, t0, lr, unroll: int):
        s0 = traj_d[case, t0]
        idx = t0[None, :] + jnp.arange(1, unroll + 1)[:, None]
        truth_seq = traj_d[case[None, :], idx]  # (unroll, B, 7, n, n)
        loss, g = jax.value_and_grad(loss_fn)(p, s0, truth_seq)
        finite = jnp.isfinite(loss) & jnp.all(jnp.asarray(
            [jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(g)]))
        p_new, opt_new, _ = adam_update(p, g, opt, lr)
        keep = lambda a, b: jnp.where(finite, a, b)  # noqa: E731
        return jax.tree_util.tree_map(keep, p_new, p), jax.tree_util.tree_map(keep, opt_new, opt), loss, finite

    stages = [tuple(int(x) for x in st.split(":")) for st in args.stages.split(",")]
    total = sum(it for _, it in stages)
    opt, rng, t0, it, skipped = adam_init(params), np.random.default_rng(0), time.perf_counter(), 0, 0
    for unroll, n_iter in stages:
        losses = []
        for _ in range(n_iter):
            lr = 1e-3 * 0.5 * (1 + np.cos(np.pi * it / total)) + 1e-5
            case = jnp.asarray(rng.integers(0, len(TRAIN_CASES), args.batch))
            start = jnp.asarray(rng.integers(0, steps - unroll, args.batch))
            params, opt, loss, finite = train_step(params, opt, case, start, lr, unroll)
            losses.append(float(loss))
            skipped += int(not bool(finite))
            it += 1
        print(f"train unroll {unroll:2d}: loss {np.nanmean(losses[-50:]):.4e} (state MSE / var) | "
              f"skipped {skipped} | {time.perf_counter() - t0:.0f} s", flush=True)
    model = {"params": params, "scale": scale, "static": static}
    mdir = ROOT / "data" / "models" / "cfd"
    mdir.mkdir(parents=True, exist_ok=True)
    np.savez(mdir / f"blast{n}_corrector.npz",
             **{f"p{i}_{k}": np.asarray(v) for i, layer in enumerate(params) for k, v in layer.items()})

    # ---- 3. schedules on the evaluation case
    solver = mhd.Solver(mhd.blast_wave(n, *EVAL_CASE), n, dt)
    s0 = solver.state0[None]
    truth_final = jnp.asarray(ev[-1])
    configs = [("none", 1, 0, 32), ("dense", 1, 0, 32)]
    for k in (int(x) for x in args.ks.split(",")):
        configs.append(("dense", k, 0, 32))
        for t in (int(x) for x in args.tile_sizes.split(",")):
            configs.append(("tiles", k, (n // t) ** 2 // k, t))

    def rel(a, b):
        return float(jnp.sqrt(jnp.sum((a - b) ** 2) / jnp.sum(b**2)))

    rows, b0_final = [], None
    for mode, k, m, t in configs:
        kw = dict(mode=mode, n_steps=steps, k=k, m=m, t=t, age_max=2 * k)
        final, _ = mhd.run_jit(solver, model, s0, **kw)
        final = final[0]
        if mode == "dense" and k == 1:
            b0_final = final
        times = []
        mhd.run_jit(solver, model, s0, **kw)[0].block_until_ready()
        for _ in range(3):
            t1 = time.perf_counter()
            mhd.run_jit(solver, model, s0, **kw)[0].block_until_ready()
            times.append(time.perf_counter() - t1)
        row = {"mode": mode, "k": k, "m": m, "tile": t, "finite": bool(jnp.all(jnp.isfinite(final))),
               "err_vs_truth": rel(final, truth_final),
               "err_vs_B0": rel(final, b0_final) if b0_final is not None else None,
               "ms_per_step": 1e3 * min(times) / steps,
               "window_spread": max(times) / min(times) - 1}
        rows.append(row)
        print(f"{mode:6s} k {k} m {m:4d} tile {t:2d} | err vs truth {row['err_vs_truth']:.4f} | vs B0 "
              f"{row['err_vs_B0'] if row['err_vs_B0'] is None else round(row['err_vs_B0'], 4)} | "
              f"{row['ms_per_step']:.3f} ms/step", flush=True)
    b0 = next(r for r in rows if r["mode"] == "dense" and r["k"] == 1)
    for r in rows:
        r["speedup_vs_B0"] = b0["ms_per_step"] / r["ms_per_step"]
    out = ROOT / "data" / "results" / "e10" / f"w2_blast_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n": n, "factor": f, "dt": dt, "steps": steps, "hidden": args.hidden,
                               "rows": rows}, indent=1))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
