"""W3: the split operator on 3D forced isotropic turbulence (32^3), trained, scored against the
classical frontier (filtered spectral at 32, 48, 64, 96 at their largest stable steps), and
run under Frost's holds.

Why 3D: in 2D the filtered spectral solver resolves so much per grid point that the learned
correction has little to add (Dresdner et al.; our W1 frontier). In 3D the spectrum falls as
k^-5/3, coarse grids miss far more, and resolution costs n^4, while the network costs a
fixed multiple of a coarse step, which Frost's hold divides by k.

Data: ``hit_dns.py`` (192^3 DNS; train/u32 and eval/u32 every 0.04 on the 32^3 band;
eval/u_ic, the initial state for the classical runs, at attr n_ic). The loss is the velocity error on
the 32^3 band over unrolled steps, divided by the plain physics step's own error on the same
windows (1.0 = no better than the base). tau is the time until the correlation of the
velocity with the DNS on the 32^3 band drops below 0.95, over the 16 eval trajectories.

Output: data/results/e10/w3split_<tag>.json; model data/models/cfd/<name>.{npz,json}

Usage (guarded launcher; GPU, JAX env)
  scripts/safe_run.sh cfd_w3 env -u LD_LIBRARY_PATH -u CUDA_HOME PYTHONPATH=src \\
      .venvs/frost-mhd/bin/python experiments/e10_cfd/w3split.py --tag w3a
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

from frost.cfd import spectral3d as s3, split3d
from frost.cfd.spectral3d import Grid3

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_closure import adam_init, adam_update  # noqa: E402
from w1_eval import RES, best_of, correlation_times  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "cfd" / "hit192.h5"
DT_C = 0.04


def save_model(path: Path, model: dict, meta: dict) -> None:
    leaves = jax.tree_util.tree_leaves(model["params"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path.with_suffix(".npz"), *[np.asarray(x) for x in leaves])
    path.with_suffix(".json").write_text(json.dumps(
        {**meta, "scale_in": model["scale_in"], "scale_out": model["scale_out"],
         "rates": list(model["static"].rates), "stochastic": bool(model["static"].stochastic)}, indent=1))


def load_model(path: Path) -> dict:
    meta = json.loads(path.with_suffix(".json").read_text())
    model = split3d.init_model(jax.random.PRNGKey(0), scale_in=meta["scale_in"], scale_out=meta["scale_out"],
                               rates=tuple(meta["rates"]), arch=meta.get("arch", "epd"), hidden=meta["hidden"],
                               blocks=meta["blocks"], base=meta.get("base", "none"), base_c=meta.get("base_c", 0.0),
                               physics=meta.get("physics", "hydro"), substeps=meta.get("substeps", 1),
                               stochastic=meta.get("stochastic", False))
    z = np.load(path.with_suffix(".npz"))
    treedef = jax.tree_util.tree_structure(model["params"])
    model["params"] = jax.tree_util.tree_unflatten(treedef, [jnp.asarray(z[f"arr_{i}"])
                                                             for i in range(treedef.num_leaves)])
    return model
    return model


@partial(jax.jit, static_argnames=("grid", "score", "dt", "n_records", "record_every"))
def physics_rollout(grid: Grid3, score: Grid3, u0, *, dt: float, n_records: int, record_every: int):
    """Classical rollout; each record keeps only the scoring grid's band (a 96^3 history is 26 GB)."""

    def record(u, _):
        u = jax.lax.fori_loop(0, record_every, lambda _, x: s3.step(grid, x, dt), u)
        return u, (u if grid == score else s3.truncate(grid, score, u))

    return jax.lax.scan(record, u0, None, length=n_records)[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", default="4:1000,8:2000,16:5000", help="unroll:iterations,...")
    ap.add_argument("--train-k", type=int, default=1)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--ema", type=float, default=0.99)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--rates", default="1,2,4,2,1")
    ap.add_argument("--scale-out", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", type=Path, default=None)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--model", type=Path, default=ROOT / "data" / "models" / "cfd" / "hit32_split")
    ap.add_argument("--ks", default="2,4,8,16")
    ap.add_argument("--cert", default="")
    ap.add_argument("--classical", default="32:1,48:1,48:2,64:2,64:3,96:3,96:4",
                    help="n:m, filtered spectral on n^3 with time step 0.04 / m")
    ap.add_argument("--batches", default="16")
    ap.add_argument("--steps", type=int, default=64, help="timed length, in steps of 0.04")
    ap.add_argument("--data", type=Path, default=DATA, help="eval/u32 and eval/u_ic (and train/u<grid>)")
    ap.add_argument("--train-data", default=None,
                    help="train/u<grid> from here instead; a comma list pools several independent DNS references "
                         "(each truncated to the shortest record count, trajectories concatenated)")
    ap.add_argument("--grid", type=int, default=32, help="the hybrid's grid (scored on the 32^3 band)")
    ap.add_argument("--arch", choices=("epd", "dw"), default="epd")
    ap.add_argument("--base", choices=("none", "cl", "smag"), default="none",
                    help="classical LES closure under the learned correction")
    ap.add_argument("--base-c", type=float, default=0.0)
    ap.add_argument("--substeps", type=int, default=1,
                    help="physics steps per 0.04 split step (MHD's Alfven speed needs 2 on 32^3); the exp filter's "
                         "per-step strength becomes 6 / substeps, so its damping per unit time is unchanged")
    ap.add_argument("--train-lag", type=int, choices=(0, 1), default=0,
                    help="1: each hold block uses the correction of the previous block's start state, as the "
                         "two-device pipeline runs it (frost.cfd.pipeline3d, lag 1), the window's first block "
                         "included (its correction comes from the DNS state one hold before the window)")
    ap.add_argument("--level-weight", type=float, default=0.0,
                    help="weight of each field's total-energy error along the window, relative to the DNS's, squared: "
                         "the MHD closure's field sat ~7-10%% low at every shell (a saturated-dynamo level offset, "
                         "not a small-scale damping) and short-window spectral losses barely see a level drift")
    ap.add_argument("--field-balance", action="store_true",
                    help="MHD: the trajectory error as the mean over fields of each field's error relative to its "
                         "own variance; plain MSE is dominated by the flow (E_kin 0.44 against E_mag 0.135) and the "
                         "closure under-serves the field")
    ap.add_argument("--skew-weight", type=float, default=0.0,
                    help="weight of a velocity-gradient skewness and flatness loss (relative squared errors "
                         "against the DNS over the same windows, on |k| <= 12): MSE and spectra do not see the "
                         "cascade's sign, which the derivative skewness measures")
    ap.add_argument("--spectral-weight", type=float, default=0.0,
                    help="weight of a shell-spectrum loss on the unrolled states (List et al. 2022): trajectory "
                         "MSE alone rewards smoothing, which biases long-run statistics low at high k")
    ap.add_argument("--stat-weight", type=float, default=0.0,
                    help="weight of an invariant-statistics term: each field's shell spectrum averaged over the "
                         "unrolled window and batch, against the long-run mean spectrum of all the training data "
                         "(not the window's own DNS spectrum). Short windows cannot see a saturated dynamo's level; "
                         "matching the pooled invariant statistics is the regulariser DySLIM (Schiff et al. 2024) uses")
    ap.add_argument("--stochastic", action="store_true",
                    help="add a noise-amplitude head (split3d.add_noise_head) and train with the energy score of two "
                         "noise samples (Gneiting and Raftery 2007) in place of the trajectory MSE; the statistics terms "
                         "average over both samples")
    ap.add_argument("--sigma-init", type=float, default=-4.0, help="--stochastic: the amplitude head's initial bias")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    with h5py.File(args.data, "r") as fh:
        attrs = dict(fh.attrs)
        ev = np.asarray(fh["eval/u32"])
        ev_ic = np.asarray(fh["eval/u_ic"])
    train = None
    if not args.skip_train:
        parts = []
        for path in (args.train_data or str(args.data)).split(","):
            with h5py.File(path, "r") as fh:
                parts.append(np.asarray(fh[f"train/u{args.grid}"]))
        n_t = min(x.shape[0] for x in parts)
        train = np.concatenate([x[:n_t] for x in parts], axis=1)
        del parts
    nu, power = float(attrs["nu"]), float(attrs["power"])
    power_b = float(attrs.get("power_b", -1.0))  # MHD: power into the field (< 0: as the flow)
    physics = str(attrs.get("physics", "hydro"))
    g = Grid3(args.grid, nu=nu, power=power, power_b=power_b, dealias="exp", filter_strength=6.0 / args.substeps)
    g32 = Grid3(32, nu=nu, power=power, power_b=power_b, dealias="exp")
    n_rec = ev.shape[0] - 1
    truth = jnp.asarray(ev[1:])
    g_ic = Grid3(int(attrs["n_ic"]), nu=nu, power=power, power_b=power_b, dealias="exp")
    w0_eval = s3.truncate(g_ic, g, s3.to_hat(g_ic, jnp.asarray(ev_ic)))
    print(f"devices {jax.devices()} | grid {g.n}^3 ({args.arch}) | train "
          f"{None if train is None else train.shape} | eval {ev.shape} | nu {nu} | train hold k {args.train_k}",
          flush=True)

    def tau_of(states, grid=g):
        st = states if grid == g32 else s3.truncate(grid, g32, states)
        rho = np.asarray(s3.correlation(s3.to_real(g32, st), truth[:, : st.shape[1]]))
        return correlation_times(np.where(np.isfinite(rho), rho, -1.0), DT_C)

    rates = tuple(int(r) for r in args.rates.split(","))
    meta = {"grid": args.grid, "arch": args.arch, "base": args.base, "base_c": args.base_c, "dealias": "exp",
            "dt": DT_C, "hidden": args.hidden, "blocks": args.blocks, "physics": physics, "substeps": args.substeps}
    if args.skip_train:
        model = load_model(args.model)
    else:
        model = (load_model(args.init) if args.init is not None else
                 split3d.init_model(jax.random.PRNGKey(args.seed), rates=rates, arch=args.arch, base=args.base,
                                    base_c=args.base_c, hidden=args.hidden, blocks=args.blocks,
                                    scale_out=args.scale_out, physics=physics, substeps=args.substeps))
        if args.stochastic and not model["static"].stochastic:
            model = split3d.add_noise_head(model, jax.random.PRNGKey(args.seed + 11), args.sigma_init)
        if args.init is not None and model["static"].substeps != args.substeps:
            # a warm start retrained for another physics rate: the step count is this run's, not the init's
            import dataclasses

            model["static"] = dataclasses.replace(model["static"], substeps=args.substeps)
        # the spectra stay in host memory (48^3: 8.8 GB, and XLA may copy a device argument);
        # each step ships only its windows
        data = np.empty((*train.shape[:-1], train.shape[-1] // 2 + 1), np.complex64)
        for t0 in range(0, train.shape[0], 25):
            data[t0:t0 + 25] = np.asarray(s3.to_hat(g, jnp.asarray(train[t0:t0 + 25])))
        del train
        kk = args.train_k
        template = {k: v for k, v in model.items() if k != "params"}
        n_shells = int(0.45 * g.n)
        shells = jnp.asarray((np.rint(np.sqrt(g.k2))[..., None] == np.arange(1, n_shells + 1)).astype(np.float32))
        spec_unit = 0.0  # set per stage to the physics step's trajectory loss, so the weight is scale-free
        stat_target = None
        if args.stat_weight:  # long-run mean shell spectra of every field over all training records and trajectories
            acc = 0.0
            for t0 in range(0, data.shape[0], 25):
                blk = jnp.asarray(data[t0:t0 + 25])
                # sum over records and trajectories first, then project on the shells (a 48^3 block's
                # (t, b, x, y, z, shells) product does not fit on the GPU)
                acc = acc + np.stack([np.asarray(jnp.einsum("xyz,xyzs->s", 0.5 * g.weight * jnp.sum(
                    jnp.abs(blk[..., f0:f0 + 3, :, :, :]) ** 2, axis=(0, 1, 2)), shells))
                    for f0 in ((0, 3) if physics == "mhd" else (0,))])
            stat_target = jnp.asarray(acc / (data.shape[0] * data.shape[1]))
            print(f"invariant-statistics target from {data.shape[1]} trajectories x {data.shape[0]} records", flush=True)
        grad_band = jnp.asarray((np.sqrt(g.k2) <= 12.5).astype(np.float32))  # stats3d.GRAD_BAND

        def traj_error(traj, targets):
            """Trajectory MSE; with --field-balance, the mean over fields of each field's MSE over its variance."""
            if not args.field_balance:
                return jnp.mean(s3.to_real(g, traj - targets) ** 2)
            errs = []
            for f0 in field_starts:
                tg = s3.to_real(g, targets[..., f0:f0 + 3, :, :, :])
                errs.append(jnp.mean((s3.to_real(g, traj[..., f0:f0 + 3, :, :, :]) - tg) ** 2) / jnp.mean(tg**2))
            return sum(errs) / len(errs)
        field_starts = (0, 3) if physics == "mhd" else (0,)

        def grad_moments(u):
            """Pooled longitudinal-derivative moments (<d^2>, <d^3>, <d^4>) of states (..., 3, n, n, n/2+1)."""
            u = u * grad_band
            acc = 0.0
            for i, ki in enumerate((g.kx, g.ky, g.kz)):
                d = jnp.fft.irfftn(1j * ki * u[..., i, :, :, :], s=(g.n,) * 3, axes=(-3, -2, -1))
                acc = acc + jnp.stack([jnp.mean(d**2), jnp.mean(d**3), jnp.mean(d**4)])
            return acc / 3.0

        def loss_fn(p, u0, targets, key=None):
            mdl = {**template, "params": p}
            noisy = key is not None and bool(mdl["static"].stochastic)

            def run(mdl, u0):
                @jax.checkpoint
                def outer(carry, _):
                    uh, c_prev, j = carry
                    c = split3d.correction(g, mdl, uh, step=j if noisy else None)
                    c_use = c_prev if args.train_lag else c  # lag 1: block n runs on the correction of block n - 1
                    uh, tr = jax.lax.scan(lambda x, _: (split3d.split_step(g, x, c_use, DT_C, split3d.base_of(mdl),
                                                                              *split3d.physics_of(mdl)),) * 2,
                                          uh, None, length=kk)
                    return (uh, c, j + 1), tr

                if args.train_lag:  # u0 = (DNS state one hold before the window, window start): the first block lags too
                    c0 = split3d.correction(g, mdl, u0[0])
                    u0 = u0[1]
                else:
                    c0 = jnp.zeros_like(u0)  # unused at lag 0: no extra network call
                _, traj = jax.lax.scan(outer, (u0, c0, jnp.zeros((), jnp.int32)), None, length=targets.shape[0] // kk)
                return traj.reshape(targets.shape)

            if noisy:
                # the energy score of two noise samples, E|X - y| - E|X - X'| / 2 in the trajectory RMS norm: proper, so
                # it rewards a calibrated spread where the MSE would drive the amplitude to zero; sqrt(unit) keeps the
                # term in the MSE's units under train_step's division by the unit
                k1, k2 = jax.random.split(key)
                trajs = [run({**mdl, "noise_key": k1}, u0), run({**mdl, "noise_key": k2}, u0)]
                rms = lambda a, b: jnp.sqrt(traj_error(a, b) + 1e-30)  # noqa: E731
                loss = (0.5 * (rms(trajs[0], targets) + rms(trajs[1], targets)) - 0.5 * rms(trajs[0], trajs[1])) \
                    * jnp.sqrt(spec_unit)
            else:  # deterministic: the trajectory MSE, then the statistics, accumulated in the original order
                traj = run(mdl, u0)
                return stats_loss(traj, targets, traj_error(traj, targets))
            return loss + sum(stats_loss(traj, targets) for traj in trajs) / len(trajs)

        def stats_loss(traj, targets, loss=0.0):
            for f0 in field_starts:  # velocity, then (MHD) the magnetic field, each with its own statistics
                tr, tg = traj[..., f0:f0 + 3, :, :, :], targets[..., f0:f0 + 3, :, :, :]
                if args.level_weight:  # the field's total energy at every step of the window, model against DNS
                    e_tr = jnp.sum(g.weight * jnp.abs(tr) ** 2, axis=(-4, -3, -2, -1))
                    e_tg = jnp.sum(g.weight * jnp.abs(tg) ** 2, axis=(-4, -3, -2, -1))
                    loss += args.level_weight * spec_unit * jnp.mean(((e_tr - e_tg) / e_tg) ** 2)
                if args.stat_weight:  # the window's mean spectrum against the pooled long-run mean (invariant)
                    e_mod = jnp.einsum("...xyz,xyzs->s", 0.5 * g.weight * jnp.sum(jnp.abs(tr) ** 2, axis=-4), shells)
                    e_mod = e_mod / (tr.shape[0] * tr.shape[1])
                    loss += args.stat_weight * spec_unit * jnp.mean(
                        (jnp.log(e_mod + 1e-30) - jnp.log(stat_target[field_starts.index(f0)] + 1e-30)) ** 2)
                if args.spectral_weight:
                    # shell spectra over the window and batch, model against DNS: (log E - log E_DNS)^2 per shell
                    e_mod = jnp.einsum("...xyz,xyzs->s", 0.5 * g.weight * jnp.sum(jnp.abs(tr) ** 2, axis=-4), shells)
                    e_dns = jnp.einsum("...xyz,xyzs->s", 0.5 * g.weight * jnp.sum(jnp.abs(tg) ** 2, axis=-4), shells)
                    loss += args.spectral_weight * spec_unit * jnp.mean(
                        (jnp.log(e_mod + 1e-30) - jnp.log(e_dns + 1e-30)) ** 2)
                if args.skew_weight:
                    mm, md = grad_moments(tr), grad_moments(tg)
                    f_m, f_d = mm[2] / mm[0] ** 2, md[2] / md[0] ** 2
                    term = ((f_m - f_d) / f_d) ** 2
                    if f0 == 0:  # b -> -b symmetry makes the field's odd moments vanish: flatness only for b
                        s_m, s_d = mm[1] / mm[0] ** 1.5, md[1] / md[0] ** 1.5
                        term = term + ((s_m - s_d) / s_d) ** 2
                    loss += args.skew_weight * spec_unit * term
            return loss

        def windows(t_idx, b_idx, unroll):
            offs = t_idx[None, :] + np.arange(1, unroll + 1)[:, None]
            start = jnp.asarray(data[t_idx, b_idx])
            if args.train_lag:  # the pipeline's first lagged block needs the state one hold earlier (t_idx >= kk)
                start = jnp.stack([jnp.asarray(data[t_idx - kk, b_idx]), start])
            return start, jnp.asarray(data[offs, b_idx[None, :]])

        @jax.jit
        def eval_loss_dev(p, u0, targets):
            return loss_fn(p, u0, targets)

        def chunked(fn, t_idx, b_idx, unroll: int, size: int = 4) -> float:
            """Mean of ``fn`` over the fixed windows in chunks of ``size`` (16 windows of 128 steps at 48^3 are ~5 GB per
            array); the chunks are equal, so the mean of chunk means is the mean."""
            parts = [float(fn(t_idx[i:i + size], b_idx[i:i + size], unroll)) for i in range(0, len(t_idx), size)]
            return sum(parts) / len(parts)

        def eval_loss(p, t_idx, b_idx, unroll: int):
            return chunked(lambda t, b, u: eval_loss_dev(p, *windows(t, b, u)), t_idx, b_idx, unroll)

        def norm_traj(t_idx, b_idx, unroll: int) -> float:
            return chunked(_norm_traj, t_idx, b_idx, unroll)

        def _norm_traj(t_idx, b_idx, unroll: int) -> float:
            """The plain physics step's trajectory MSE on fixed windows (the loss unit)."""
            u0, targets = windows(t_idx, b_idx, unroll)
            u0 = u0[1] if args.train_lag else u0
            traj = jax.lax.scan(lambda x, _: (split3d.physics_step(g, x, DT_C, ("none", 0.0), physics,
                                                                   args.substeps),) * 2, u0, None, length=unroll)[1]
            return float(traj_error(traj, targets))

        @jax.jit
        def train_step(p, opt, ema, u0, targets, lr, norm, key=None):
            loss, grads = jax.value_and_grad(lambda q: loss_fn(q, u0, targets, key) / norm)(p)
            finite = jnp.isfinite(loss) & jnp.all(jnp.asarray(
                [jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(grads)]))
            p_new, opt_new, _ = adam_update(p, grads, opt, lr, b2=0.99)
            keep = lambda a, b: jnp.where(finite, a, b)  # noqa: E731
            p = jax.tree_util.tree_map(keep, p_new, p)
            opt = jax.tree_util.tree_map(keep, opt_new, opt)
            ema = jax.tree_util.tree_map(lambda e, q: args.ema * e + (1 - args.ema) * q, ema, p)
            return p, opt, ema, loss, finite

        def quick_tau(p):
            st = split3d.rollout(g, {**template, "params": p}, w0_eval[:4], k=1, dt=DT_C, n_records=n_rec,
                                 record_every=1)
            return float(tau_of(st).mean())

        stages = [tuple(int(x) for x in s.split(":")) for s in args.stages.split(",")]
        total = sum(n for _, n in stages)
        params = model["params"]
        opt, ema = adam_init(params), params
        zero_p = jax.tree_util.tree_map(jnp.zeros_like, params)
        rng, it, skipped, last_skipped, t_start = np.random.default_rng(args.seed), 0, 0, 0, time.perf_counter()
        print(f"start quick tau {quick_tau(params):.2f} | {args.arch} {split3d.epd3_flops(params, g.n) / 1e9:.1f} GFLOP "
              f"per call | rates {rates}", flush=True)
        for unroll, n_iter in stages:
            t_max = data.shape[0] - 1 - unroll
            t_min = kk if args.train_lag else 0
            fixed = np.random.default_rng(123)
            tf = fixed.integers(t_min, t_max + 1, 16)
            bf = fixed.integers(0, data.shape[1], 16)
            norm = (float(eval_loss(zero_p, tf, bf, unroll))
                    if not (args.spectral_weight or args.skew_weight or args.field_balance or args.level_weight)
                    else norm_traj(tf, bf, unroll))
            spec_unit = norm
            print(f"stage unroll {unroll}: physics-step loss {norm:.4e} (the unit); current model "
                  f"{float(eval_loss(params, tf, bf, unroll)) / norm:.4f}", flush=True)
            for _ in range(n_iter):
                lr = args.lr * 0.5 * (1 + math.cos(math.pi * it / total)) + 1e-6
                lr *= min(1.0, (it + 1) / args.warmup) if args.warmup else 1.0
                t_idx = rng.integers(t_min, t_max + 1, args.batch)
                b_idx = rng.integers(0, data.shape[1], args.batch)
                key = jax.random.PRNGKey(args.seed * 7919 + it) if model["static"].stochastic else None
                params, opt, ema, loss, finite = train_step(params, opt, ema, *windows(t_idx, b_idx, unroll), lr,
                                                            norm, key)
                it += 1
                skipped += int(not bool(finite))
                if it % 250 == 0:
                    print(f"it {it:6d} unroll {unroll:3d} loss {float(loss):.4e} skipped {skipped} | "
                          f"{time.perf_counter() - t_start:.0f} s", flush=True)
                    if skipped - last_skipped > 225:
                        raise SystemExit(f"training stuck: {skipped - last_skipped} of the last 250 non-finite")
                    last_skipped = skipped
                if it % 2000 == 0:
                    print(f"  quick tau (4 eval runs) {quick_tau(ema):.3f} (EMA)", flush=True)
            print(f"stage unroll {unroll} done: quick tau {quick_tau(ema):.3f} (EMA) | fixed windows "
                  f"{float(eval_loss(ema, tf, bf, unroll)) / norm:.4f} of the physics step", flush=True)
        model = {**template, "params": ema}
        save_model(args.model, model, {**meta, "stages": args.stages, "train_k": kk, "skipped": skipped,
                                       "train_seconds": time.perf_counter() - t_start, "lr": args.lr})
        del data

    # ---------------------------------------------------------------- scores
    rows: dict[str, dict] = {}

    def score(name, states, grid=g, extra=None):
        tau = tau_of(states, grid)
        rows[name] = {"tau": tau.tolist(), "tau_mean": float(tau.mean()),
                      "tau_sem": float(tau.std(ddof=1) / np.sqrt(len(tau))),
                      "finite": bool(jnp.all(jnp.isfinite(states.real))), **(extra or {})}
        print(f"{name:24s} tau {tau.mean():6.3f} +- {rows[name]['tau_sem']:.3f} | finite {rows[name]['finite']}"
              + (f" | {extra}" if extra else ""), flush=True)

    def classical(n):
        if n > g_ic.n:
            raise SystemExit(f"classical n {n} exceeds the stored initial state ({g_ic.n}^3)")
        gc = Grid3(n, nu=nu, power=power, power_b=power_b, dealias="exp")
        return gc, s3.truncate(g_ic, gc, s3.to_hat(g_ic, jnp.asarray(ev_ic)))

    ks = [int(x) for x in args.ks.split(",") if x]
    cert = [(int(c.split(":")[0]), float(c.split(":")[1])) for c in args.cert.split(",") if c]
    for k in [1] + ks:
        score(f"split:value:{k}", split3d.rollout(g, model, w0_eval, k=k, dt=DT_C, n_records=n_rec, record_every=1))
    for k_c, e_c in cert:
        st, n_rb, e_mean = split3d.cert_rollout(g, model, w0_eval, k=k_c, dt=DT_C, e_star=e_c, n_records=n_rec,
                                                record_every=1)
        score(f"split:cert:{k_c}:{e_c}", st, extra={"rollbacks": int(n_rb), "intervals": n_rec // k_c,
                                                    "mean_hold_error": float(e_mean)})
    classical_runs = [tuple(int(x) for x in c.split(":")) for c in args.classical.split(",") if c]
    for n, m in classical_runs:
        gc, u0 = classical(n)
        score(f"spectral{n}exp:dt/{m}", physics_rollout(gc, g32, u0, dt=DT_C / m, n_records=n_rec, record_every=m),
              grid=g32)
    base = np.asarray(rows["split:value:1"]["tau"])
    for r in rows.values():
        d = np.asarray(r["tau"]) - base
        r["dtau_vs_B0"], r["dtau_sem"] = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))

    # ---------------------------------------------------------------- wall clock
    timing = {}
    for b in (int(x) for x in args.batches.split(",")):
        tile = lambda w: jnp.tile(w, (-(-b // w.shape[0]), 1, 1, 1, 1))[:b]  # noqa: E731
        wb = tile(w0_eval)
        t, windows = {}, {}

        def timed(name, fn):
            lo, hi = best_of(fn)
            t[name], windows[name] = 1e3 * lo / args.steps, lo / hi

        net = jax.jit(lambda w: split3d.correction(g, model, w))
        timed("network_call", lambda: net(wb))
        t["network_call"] *= args.steps
        for k in [1] + ks:
            timed(f"split:value:{k}", lambda: split3d.advance(g, model, wb, k=k, dt=DT_C, n_steps=args.steps))
        for k_c, e_c in cert:
            timed(f"split:cert:{k_c}:{e_c}", lambda: split3d.cert_advance(g, model, wb, k=k_c, dt=DT_C, e_star=e_c,
                                                                          n_steps=args.steps))
        for n, m in classical_runs:
            gc, u0 = classical(n)
            uc = tile(u0)
            stepper = jax.jit(lambda u, gc=gc, m=m: jax.lax.fori_loop(
                0, args.steps * m, lambda _, x: s3.step(gc, x, DT_C / m), u))
            timed(f"spectral{n}exp:dt/{m}", lambda: stepper(uc))
        timing[str(b)] = {"ms_per_0.04": t, "window_ratio": windows,
                          "speedup_vs_B0": {n: t["split:value:1"] / v for n, v in t.items() if n != "network_call"}}
        print(f"batch {b}: " + " | ".join(f"{n} {v:.2f}" for n, v in t.items()) + " ms per 0.04 "
              "(network_call: ms per call)", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    out = RES / f"w3split_{args.tag}.json"
    out.write_text(json.dumps({"accuracy": rows, "timing": timing, "model": str(args.model)}, indent=1))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
