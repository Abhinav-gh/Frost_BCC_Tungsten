"""Train the learned closure for workload W1 through the solver (new_directions.md S3, test 11).

Solver-in-the-loop training, as in Kochkov et al. 2021 and Um et al. 2020: start
from a filtered-DNS state on 64^2, unroll the coarse solver with the closure applied
every step (the ``operator`` form at k = 1, i.e. B0), and penalise the vorticity
error against the filtered DNS at every step.

The unroll grows over three stages (4, 16, 32 steps), with gradient clipping and
a cosine learning rate. Every step is rematerialised (``jax.checkpoint``), so
memory is one step's activations per unroll step.

The network has six 3x3 layers of 64 channels (GELU). By default it outputs the
structured operator of ``frost.cfd.closure``: a PSD eddy-diffusivity tensor plus an
enstrophy-conserving stream function (``--form tensor``). The unconstrained 18-
coefficient stencil (``--form stencil``) blew up within 16-step unrolls on
2026-10-01, and one NaN batch poisoned every weight. So an update whose loss or
gradient is non-finite is now skipped and counted, never applied.

Adam is written here: the env has no optax, and it is twenty lines.

Output: data/models/cfd/kolmogorov64_flux_cnn.npz (+ .json)

Usage (guarded launcher; GPU, JAX env)
  scripts/safe_run.sh cfd_train env -u LD_LIBRARY_PATH -u CUDA_HOME PYTHONPATH=src \\
      .venvs/frost-mhd/bin/python experiments/e10_cfd/train_closure.py
"""

from __future__ import annotations

import argparse
import json
import math
import time
from functools import partial
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from frost.cfd import closure, schedule, spectral
from frost.cfd.spectral import Grid

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "cfd" / "kolmogorov_re1000.h5"
OUT = ROOT / "data" / "models" / "cfd" / "kolmogorov64_flux_cnn"
DT = 0.007


def save_model(path: Path, model: dict, meta: dict) -> None:
    leaves = {f"p{i}_{name}": np.asarray(v) for i, layer in enumerate(model["params"])
              for name, v in layer.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path.with_suffix(".npz"), **leaves)
    path.with_suffix(".json").write_text(json.dumps({**meta, "scale": model["scale"],
                                                      "layers": len(model["params"])}, indent=1))


def load_model(path: Path) -> dict:
    meta = json.loads(path.with_suffix(".json").read_text())
    z = np.load(path.with_suffix(".npz"))
    params = [{"w": jnp.asarray(z[f"p{i}_w"]), "b": jnp.asarray(z[f"p{i}_b"])}
              for i in range(meta["layers"])]
    return {"params": params, "scale": meta["scale"]}


def adam_init(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": jnp.zeros((), jnp.int32)}


def adam_update(params, grads, state, lr, b1=0.9, b2=0.999, eps=1e-8, clip=1.0):
    norm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree_util.tree_leaves(grads)))
    grads = jax.tree_util.tree_map(lambda g: g * jnp.minimum(1.0, clip / (norm + 1e-12)), grads)
    t = state["t"] + 1
    m = jax.tree_util.tree_map(lambda a, g: b1 * a + (1 - b1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda a, g: b2 * a + (1 - b2) * g * g, state["v"], grads)
    c1, c2 = 1 - b1**t, 1 - b2**t
    params = jax.tree_util.tree_map(lambda p, a, b: p - lr * (a / c1) / (jnp.sqrt(b / c2) + eps),
                                    params, m, v)
    return params, {"m": m, "v": v, "t": t}, norm


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", default="4:400,16:800,32:1200", help="unroll:iterations,...")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--form", choices=("tensor", "stencil"), default="tensor")
    ap.add_argument("--k-scale", type=float, default=5e-3, help="eddy diffusivity unit (5 nu)")
    ap.add_argument("--s-scale", type=float, default=0.1, help="subgrid stream-function unit")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    g = Grid(64)
    with h5py.File(args.data, "r") as fh:
        train = np.asarray(fh["train/w64"])  # (T, B, 64, 64), one per 64^2 step
        evals = np.asarray(fh["eval/w64"][:, :4])  # (R, 4, 64, 64), every 16 steps
    t_steps, n_traj = train.shape[:2]
    w_hat_all = jnp.asarray(np.fft.rfft2(train) * g.mask, jnp.complex64)
    u, v = spectral.velocity(g, w_hat_all[::50])
    scale = {"u": float(jnp.sqrt(jnp.mean(u**2 + v**2) / 2)), "w": float(np.sqrt(np.mean(train**2))),
             "k": args.k_scale, "s": args.s_scale}
    print(f"train {train.shape} | scales {scale} | devices {jax.devices()}", flush=True)

    params = closure.init_cnn(jax.random.PRNGKey(args.seed), hidden=args.hidden, layers=args.layers,
                              out_ch=closure.N_RAW[args.form])
    print(f"CNN: {args.layers} x {args.hidden} ch, {closure.cnn_flops(params, 64) / 1e9:.2f} GFLOP "
          f"per call on 64^2", flush=True)
    ref_norm = float(np.mean(train**2))

    def loss_fn(p, w0, targets):
        model = {"params": p, "scale": scale}

        @jax.checkpoint
        def body(wh, _):
            coef = closure.coefficients(g, model, wh)
            wh = spectral.step(g, wh, DT, lambda x: closure.apply_operator(g, coef, x))
            return wh, spectral.to_real(g, wh)

        _, traj = jax.lax.scan(body, w0, None, length=targets.shape[0])
        return jnp.mean((traj - targets) ** 2) / ref_norm

    @partial(jax.jit, static_argnames=("unroll",))
    def train_step(p, opt, data, data_hat, idx_t, idx_b, lr, unroll: int):
        # the data are arguments, not closed-over constants: a closed-over array is baked
        # into the executable
        w0 = data_hat[idx_t, idx_b]
        offs = idx_t[None, :] + jnp.arange(1, unroll + 1)[:, None]
        targets = data[offs, idx_b[None, :]]
        loss, grads = jax.value_and_grad(loss_fn)(p, w0, targets)
        finite = jnp.isfinite(loss) & jnp.all(jnp.asarray(
            [jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads)]))
        p_new, opt_new, norm = adam_update(p, grads, opt, lr)
        keep = lambda new, old: jnp.where(finite, new, old)  # noqa: E731
        p = jax.tree_util.tree_map(keep, p_new, p)
        opt = jax.tree_util.tree_map(keep, opt_new, opt)
        return p, opt, loss, norm, finite

    train_dev = jnp.asarray(train)

    def correlation_time(p):
        model = {"params": p, "scale": scale}
        w0 = jnp.asarray(np.fft.rfft2(evals[0]) * g.mask, jnp.complex64)
        states = schedule.rollout(g, model, w0, mode="operator", k=1, dt=DT,
                                  n_records=evals.shape[0] - 1, record_every=16)
        rho = np.asarray(spectral.correlation(spectral.to_real(g, states), jnp.asarray(evals[1:])))
        rho = np.where(np.isfinite(rho), rho, -1.0)  # a blow-up decorrelates; NaN < 0.95 is False
        below = (rho < 0.95).argmax(axis=0)
        below = np.where((rho < 0.95).any(axis=0), below, rho.shape[0])
        return float(np.mean(below + 1) * 16 * DT)

    opt = adam_init(params)
    rng = np.random.default_rng(args.seed)
    stages = [tuple(int(x) for x in s.split(":")) for s in args.stages.split(",")]
    total = sum(n for _, n in stages)
    it, t0, history, skipped = 0, time.perf_counter(), [], 0
    print(f"untrained (no closure) correlation time {correlation_time(params):.2f}", flush=True)
    for unroll, n_iter in stages:
        for _ in range(n_iter):
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * it / total)) + 1e-5
            idx_t = jnp.asarray(rng.integers(0, t_steps - unroll - 1, args.batch))
            idx_b = jnp.asarray(rng.integers(0, n_traj, args.batch))
            params, opt, loss, norm, finite = train_step(params, opt, train_dev, w_hat_all, idx_t,
                                                         idx_b, lr, unroll)
            it += 1
            skipped += int(not bool(finite))
            if it % 100 == 0:
                history.append((it, unroll, float(loss)))
                print(f"it {it:5d} unroll {unroll:2d} loss {float(loss):.4e} grad {float(norm):.2e} "
                      f"lr {lr:.1e} | skipped {skipped} | {time.perf_counter() - t0:.0f} s", flush=True)
        print(f"stage unroll {unroll} done: correlation time (4 eval runs) "
              f"{correlation_time(params):.2f}", flush=True)
    meta = {"grid": 64, "dt": DT, "hidden": args.hidden, "stages": args.stages, "form": args.form,
            "skipped_updates": skipped,
            "batch": args.batch, "lr": args.lr, "seed": args.seed, "history": history,
            "train_seconds": time.perf_counter() - t0,
            "gflop_per_call": closure.cnn_flops(params, 64) / 1e9}
    save_model(args.out, {"params": params, "scale": scale}, meta)
    print(f"wrote {args.out}.npz", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
