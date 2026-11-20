"""G1, schedule-family fine-tuning: one NeuralGCM physics for several dynamical-core rates.

``forecast_skill.py`` showed the dynamical core's rate unlocks once its per-substep filters
keep their damping per hour (``subS+damp``): 3 and 2 substeps (20 and 30 min) keep the
released model's skill from day 2 on, at 1.40x and 1.75x, but add 0.6% and 3.3% to the
day-1 z500 RMSE. That residue is the core's time-discretisation error, which the learned
physics was trained to compensate at 12 minutes and nowhere else.

This fine-tunes only the learned physics (``stochastic_physics_parameterization_step``,
4.1M of the model's 14.5M parameters; encoder and decoder stay as released) so that one
parameter set serves a family of schedules ``--family``, written as ``forecast_skill.py``
variants: substep counts with ``damp``, float32 transforms, and physics holds (``hold2``:
the tendency reused for two intervals, which the released physics cannot take). The target is the released model itself, run at its own schedule, as
MD's students are trained against their teacher: from a pool of start states, the released
model's hourly states over ``--horizons`` (growing) are the labels, and the loss is the squared
difference of every prognostic field after each hour, normalised per field and level by
the released model's own hourly change. The substep count is drawn per step, so the
released schedule stays in the family and its skill is held by the same loss.

Start states: ERA5 2019 (``era5_prep.py --parts init --tag train2019``; NeuralGCM was
trained on 1979-2017, and G1 tests on 2020), encoded, then advanced by the released model
and sampled every ``--pool-every`` hours, so the pool covers both forecast starts and
states on the model's own trajectories. Starts after ``--val-from`` are held out.

Output: data/gcm/models/tuned/<tag>.pkl (a checkpoint dict, the released one with the
physics parameters replaced) and data/results/e11/finetune_<tag>.json.

Usage (guarded launcher; GPU, frost-gcm env)
  scripts/safe_run.sh gcm_finetune env -u LD_LIBRARY_PATH -u CUDA_HOME PYTHONPATH=src \\
      .venvs/frost-gcm/bin/python experiments/e11_gcm/schedule_finetune.py --tag hold2
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import xarray

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast_skill import parse_variant

from frost.gcm.neuralgcm_split import SplitModel

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "gcm"
RES = ROOT / "data" / "results" / "e11"
PHYSICS = "stochastic_physics_parameterization_step"


def fields(x):
    """The prognostic fields of a model state, as a flat list of [level, m, l] arrays."""
    s = x.state
    return [s.vorticity, s.divergence, s.temperature_variation, s.log_surface_pressure,
            *(s.tracers[k] for k in sorted(s.tracers))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=DATA / "models" / "deterministic_2_8_deg.pkl")
    ap.add_argument("--data-tag", default="train2019")
    ap.add_argument("--family", default="released;sub2+damp+f32;sub2+damp+f32+hold2",
                    help="schedules as forecast_skill.py variants, semicolon-separated (a bare integer S means subS+damp)")
    ap.add_argument("--horizons", default="3,6,12,24",
                    help="hours of the unrolled loss, in equal shares of the steps, short first: a held-physics "
                         "rollout diverges within 6 h before training, so it is learned from short horizons up")
    ap.add_argument("--pool-every", type=int, default=12, help="hours between pooled states")
    ap.add_argument("--pool-days", type=float, default=4.0, help="released-model run per start for the pool")
    ap.add_argument("--val-from", default="2019-11-01")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    with open(args.model, "rb") as fh:
        ckpt = pickle.load(fh)
    family = [f"sub{v}+damp" if v.isdigit() else v for v in args.family.replace(",", ";").split(";")]
    teacher = SplitModel(ckpt)
    if teacher.num_substeps != 1:
        raise ValueError("one physics interval per model step assumed (NeuralGCM-2.8)")
    students = {}
    for v in family:
        overrides, hold, _ = parse_variant(v, ckpt["model_config_str"])
        students[v] = (SplitModel(ckpt, overrides), hold)
    params = teacher.params
    train_keys = sorted(k for k in params if PHYSICS in k)
    trainable = {k: params[k] for k in train_keys}
    frozen = {k: v for k, v in params.items() if k not in trainable}
    n_train = sum(int(np.size(v)) for mod in trainable.values() for v in mod.values())
    print(f"devices {jax.devices()} | family {family} | {n_train / 1e6:.2f}M trainable of "
          f"{sum(int(np.size(v)) for mod in params.values() for v in mod.values()) / 1e6:.2f}M", flush=True)

    # --- start states -------------------------------------------------------------------
    ds = xarray.open_dataset(DATA / f"era5_128x64_{args.data_tag}_init.nc")
    times = ds.time.values
    val_mask = times >= np.datetime64(args.val_from)

    pool_len = round(args.pool_days * 24 / args.pool_every)

    @jax.jit
    def pool_from(x, forcings):
        def block(x, _):
            x = jax.lax.fori_loop(0, args.pool_every, lambda i, c: teacher.advance(
                params, c, teacher.forcing_at(params, forcings, c)), x)
            return x, x
        return jax.lax.scan(block, x, None, length=pool_len)[1]

    pool, pool_val, forcings_of = [], [], []
    t0 = time.perf_counter()
    for i in range(len(times)):
        d = ds.isel(time=i)
        forcings = teacher.api.forcings_from_xarray(d)
        x0 = teacher.api.encode(teacher.api.inputs_from_xarray(d), forcings, jax.random.key(0))
        traj = pool_from(x0, forcings)
        states = [x0] + [jax.tree_util.tree_map(lambda a, j=j: a[j], traj) for j in range(pool_len)]
        forcings_of.append(forcings)
        (pool_val if val_mask[i] else pool).extend((i, s) for s in states)  # ~13 MB each, kept on the GPU
    print(f"pool: {len(pool)} training and {len(pool_val)} held-out states from {len(times)} starts "
          f"({val_mask.sum()} held out), {time.perf_counter() - t0:.0f} s", flush=True)

    # --- labels and normalisation -------------------------------------------------------
    @partial(jax.jit, static_argnames=("hours",))
    def released_hours(x, forcings, hours: int):
        def hour(c, _):
            c = teacher.advance(params, c, teacher.forcing_at(params, forcings, c))
            return c, fields(c)
        return jax.lax.scan(hour, x, None, length=hours)[1]  # list of [hours, level, m, l]

    # per field and level: RMS of the released model's hourly change, over a sample of the pool
    rng = np.random.default_rng(args.seed)
    sq = None
    for j in rng.choice(len(pool), size=min(24, len(pool)), replace=False):
        i, x = pool[j]
        ys = released_hours(x, forcings_of[i], 24)
        prev = [f[None] for f in fields(x)]
        d = [jnp.mean((jnp.concatenate([p, y[:-1]]) - y) ** 2, axis=(0, 2, 3)) for p, y in zip(prev, ys, strict=True)]
        sq = d if sq is None else [a + b for a, b in zip(sq, d, strict=True)]
    scale = [jnp.sqrt(s / 24)[:, None, None] + 1e-30 for s in sq]

    def make_loss(s, hours: int):
        m, hold = students[s]

        def loss(train, x, forcings, ys):
            p = {**frozen, **train}

            @jax.checkpoint
            def hour(carry, j):  # physics refreshed every ``hold`` intervals, held in between
                c, held = carry
                f = m.forcing_at(p, forcings, c)
                if hold == 1:
                    held = m.physics(p, c, f)
                else:
                    held = jax.lax.cond(j % hold == 0, lambda: m.physics(p, c, f), lambda: held)
                c = m.finish(p, c, held, f)
                return (c, held), fields(c)

            held0 = jax.tree_util.tree_map(jnp.zeros_like, jax.eval_shape(
                lambda: m.physics(p, x, m.forcing_at(p, forcings, x))))
            out = jax.lax.scan(hour, (x, held0), jnp.arange(hours))[1]
            per = [jnp.mean(((a - b) / sc) ** 2) for a, b, sc in zip(out, ys, scale, strict=True)]
            return sum(per) / len(per)
        return loss

    warmup = max(1, min(50, args.steps // 5))
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(optax.warmup_cosine_decay_schedule(
        0.0, args.lr, warmup, max(args.steps, warmup + 1), args.lr * 0.1)))
    opt_state = opt.init(trainable)
    horizons = [int(h) for h in args.horizons.split(",")]
    steps = {}

    def step_fn(s, hours):
        """(train step, loss) for schedule ``s`` over ``hours``, compiled on first use. A step whose loss
        or gradient is not finite (a held-physics rollout that diverged) leaves the parameters alone."""
        if (s, hours) not in steps:
            lf = make_loss(s, hours)

            @jax.jit
            def step(train, opt_state, x, forcings, ys):
                value, grad = jax.value_and_grad(lf)(train, x, forcings, ys)
                finite = jnp.isfinite(value) & jnp.all(jnp.asarray(
                    [jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grad)]))
                grad = jax.tree_util.tree_map(lambda g: jnp.where(finite, g, 0.0), grad)
                upd, new_state = opt.update(grad, opt_state, train)
                new_train = optax.apply_updates(train, upd)
                keep = lambda a, b: jnp.where(finite, a, b)  # noqa: E731
                return (jax.tree_util.tree_map(keep, new_train, train),
                        jax.tree_util.tree_map(keep, new_state, opt_state), value, finite)
            steps[(s, hours)] = (step, jax.jit(lf))
        return steps[(s, hours)]

    def validate(train, hours):
        out = {}
        for s in family:
            vals = []
            for i, x in pool_val[:: max(1, len(pool_val) // 12)]:
                vals.append(float(step_fn(s, hours)[1](train, x, forcings_of[i], released_hours(x, forcings_of[i], hours))))
            out[s] = float(np.mean(vals))
        return out

    log = {"args": {k: str(v) for k, v in vars(args).items()}, "val": [], "train": []}
    for h in sorted({horizons[0], horizons[-1]}):
        v0 = validate(trainable, h)
        log["val"].append({"step": 0, "hours": h, **{str(k): v for k, v in v0.items()}})
        print(f"held-out loss before ({h} h): " + " | ".join(f"{s} {v:.4f}" for s, v in v0.items()), flush=True)
    t0 = time.perf_counter()
    running = {s: [] for s in family}
    skipped = 0
    out = DATA / "models" / "tuned" / f"{args.tag}.pkl"
    for k in range(1, args.steps + 1):
        hours = horizons[min(len(horizons) - 1, (k - 1) * len(horizons) // args.steps)]  # equal shares, short first
        s = family[rng.integers(len(family))]
        i, x = pool[rng.integers(len(pool))]
        ys = released_hours(x, forcings_of[i], hours)
        trainable, opt_state, value, finite = step_fn(s, hours)[0](trainable, opt_state, x, forcings_of[i], ys)
        if bool(finite):
            running[s].append(float(value))
        else:
            skipped += 1
        if k % 50 == 0:
            means = {s: float(np.mean(v[-20:])) if v else float("nan") for s, v in running.items()}
            log["train"].append({"step": k, "hours": hours, "skipped": skipped, **{str(s): v for s, v in means.items()}})
            print(f"step {k:5d} {time.perf_counter() - t0:6.0f} s | {hours} h | skipped {skipped} | train " +
                  " | ".join(f"{s}: {v:.4f}" for s, v in means.items()), flush=True)
        if k % 300 == 0 or k == args.steps:
            v = validate(trainable, hours)
            log["val"].append({"step": k, "hours": hours, **{str(s): x for s, x in v.items()}})
            print(f"held-out loss at {k} ({hours} h): " + " | ".join(f"{s} {x:.4f}" for s, x in v.items()), flush=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".tmp")
            with open(tmp, "wb") as fh:
                pickle.dump({**ckpt, "params": {**frozen, **jax.device_get(trainable)}}, fh)
            tmp.replace(out)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"finetune_{args.tag}.json").write_text(json.dumps(log, indent=1))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
