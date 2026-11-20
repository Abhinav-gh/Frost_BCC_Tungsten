"""G1 gate: do Frost's execution variants keep NeuralGCM's forecast skill?

The GCM counterpart of MD's G3: run the released model and each variant from the same ERA5
initial states (January 2020, held out of NeuralGCM's training), score latitude-weighted
RMSE of z500 and t850 against ERA5 every 6 hours to ``--days``, and judge each variant's
change in RMSE against the released model's own noise floor: the spread of RMSE across
members started from inputs perturbed at the scale of float32 reduction-order differences
(``--roundoff``, 1e-6 relative: 1e-7 is below float32's epsilon and leaves most inputs
unchanged, which made the first floor 0.13 m2/s2 at day 10). A variant is
skill-equivalent at a lead if its mean RMSE change lies within two standard errors of the
round-off members' changes, or improves.

That floor is a sensitivity test, not a skill test: by day 10 chaos has amplified any change
of the numerics (FFT for DFT moved day-10 z500 RMSE by +0.2%) far past round-off. Skill
itself is judged the way forecast centres judge a model change: a paired test over the
initial times. Per lead, the variant's RMSE minus the released model's from the same start,
with a 95% interval (Student t; the sample size deflated by the lag-1 correlation between
consecutive starts, whose forecasts overlap). A variant is **certified** at margin
``--margin`` (relative, default 1%) if the interval's upper end stays below margin x the
released RMSE at every lead, for every scored field: a non-inferiority test, the margin
playing the part of MD's force budget.

Variants (``--variants``, semicolon-separated):
- ``released``: ``PressureLevelModel``'s step as configured;
- ``fft``: the dynamical core's longitude transform by FFT (``frost.gcm.transforms``);
- ``f32``: the released transform with its einsums in native float32, not 6-pass bfloat16;
- ``holdK``: the learned physics refreshed every K physics intervals and held in between
  (the released model refreshes every interval: 60 min at 2.8 degrees, 30 at 1.4);
- ``subS``: S dynamical-core substeps per physics interval instead of the released 5 (a
  longer dycore step: 60/S minutes at 2.8 degrees), set through the config's own macro;
- ``damp``: with ``subS``, keep the dycore filter's damping per hour at the released value.
  ``DycoreWithPhysicsCorrector`` builds its exponential filter with the model step (1 h)
  as ``dt`` but applies it after every substep, so each application damps by dt/tau = 0.5
  at the cutoff and an hour by S x 0.5: fewer substeps silently weaken the filter (3
  substeps keep 60% of the trained damping). ``damp`` sets ``DYCORE_TAU`` to tau x S/S0,
  which restores the product of the per-substep factors exactly;
- ``stab``: with ``damp``, do the same for the dycore's second per-substep filter, the
  sharp stability filter (``stability/ExponentialFilter``: tau 4 min, cutoff 0.4, order
  10 at 2.8 degrees). The physics output filter shares that configurable, so the dycore
  gets its own scoped copy (``dycorestab/``) and the physics filter stays as released;
- ``tuned``: the parameters of ``--tuned`` (a checkpoint from ``schedule_finetune.py``)
  instead of the released ones;
- combinations with ``+`` (``sub3+hold2``).
Forcings (SST, sea ice) are persisted from the initial time, as in NeuralGCM's demo.
Wall clock per model step is measured inside the same ``lax.scan`` (best of three).

Output: data/results/e11/skill_<tag>.json

Usage (guarded launcher; GPU, frost-gcm env)
  scripts/safe_run.sh gcm_skill env -u LD_LIBRARY_PATH -u CUDA_HOME PYTHONPATH=src \\
      .venvs/frost-gcm/bin/python experiments/e11_gcm/forecast_skill.py --model data/gcm/models/deterministic_2_8_deg.pkl --tag 2p8
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats
import xarray

from frost.gcm.neuralgcm_split import SplitModel, f32_transforms, fft_transforms

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "data" / "results" / "e11"
DATA = ROOT / "data" / "gcm"
SCORED = (("geopotential", 500), ("temperature", 850))


def released_macro(config: str, binding: str) -> str:
    """A gin macro's or binding's value as the checkpoint's own config sets it."""
    return re.search(rf"^{re.escape(binding)} = (.+)$", config, re.M).group(1).strip()


def paired_interval(d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, 95% half-width and effective sample size of paired differences ``d`` [init, lead].

    Consecutive starts share most of their verification window, so the sample size is
    deflated by the lag-1 correlation of the differences across starts, n (1 - r) / (1 + r).
    """
    n = d.shape[0]
    mean = d.mean(0)
    if n < 3:
        return mean, np.full_like(mean, np.inf), np.full_like(mean, float(n))
    dc = d - mean
    r1 = np.clip((dc[1:] * dc[:-1]).sum(0) / np.maximum((dc ** 2).sum(0), 1e-30), 0.0, 0.95)
    n_eff = np.maximum(n * (1.0 - r1) / (1.0 + r1), 2.0)
    half = scipy.stats.t.ppf(0.975, n_eff - 1.0) * d.std(0, ddof=1) / np.sqrt(n_eff)
    return mean, half, n_eff


def _tendency_like(tend, state):
    """``state``'s fields in the physics tendency's structure (the fields the tendency carries)."""
    import dataclasses

    return dataclasses.replace(tend, **{f.name: (None if getattr(tend, f.name) is None else getattr(state, f.name))
                                        for f in dataclasses.fields(tend)})


def _relaxation_slope(t_new, t_old, s_new, s_old):
    """Per field and level, the least-squares slope of the tendency against the state over the last
    refresh interval, clipped to <= 0 (damping): the held physics then relaxes the *current* state
    instead of applying a frozen push (the CFD operator hold, at interval granularity)."""
    def slope(tn, to, sn, so):
        ds, dt = sn - so, tn - to
        axes = tuple(range(1, ds.ndim))
        num = jnp.sum(dt * ds, axis=axes, keepdims=True)
        den = jnp.sum(ds * ds, axis=axes, keepdims=True)
        return jnp.minimum(jnp.where(den > 0, num / jnp.maximum(den, 1e-30), 0.0), 0.0)

    return jax.tree_util.tree_map(slope, t_new, t_old, s_new, s_old)


def parse_variant(name: str, config: str) -> tuple[str, int, bool]:
    overrides, hold, substeps, damp, stab, relax = [], 1, None, False, False, False
    for part in name.split("+"):
        if part == "fft":
            overrides.append(fft_transforms())
        elif part == "f32":
            overrides.append(f32_transforms())
        elif part.startswith("hold"):
            hold = int(part[4:])
        elif part.startswith("relax"):  # physics every K intervals, a fitted relaxation in between
            hold, relax = int(part[5:]), True
        elif part.startswith("sub"):  # dynamical-core substeps per physics interval (released: 5)
            substeps = int(part[3:])
            overrides.append(f"N_INNER_DYCORE_STEPS = {substeps}")
        elif part == "damp":
            damp = True
        elif part == "stab":
            stab = True
        elif part == "tuned":
            pass  # parameters, not configuration: chosen in main
        elif part != "released":
            raise ValueError(part)
    if damp:
        if substeps is None:
            raise ValueError("damp needs subS")
        s0 = int(released_macro(config, "N_INNER_DYCORE_STEPS"))
        value, unit = released_macro(config, "DYCORE_TAU").strip("'\"").split()
        overrides.append(f"DYCORE_TAU = '{float(value) * substeps / s0} {unit}'")
        if stab:
            value, unit = released_macro(config, "STABILITY_TAU").strip("'\"").split()
            overrides += ["dycore/SequentialStepFilter.filter_modules = (@dycore/ExponentialFilter, "
                          "@dycorestab/ExponentialFilter)",
                          f"dycorestab/ExponentialFilter.cutoff = {released_macro(config, 'stability/ExponentialFilter.cutoff')}",
                          f"dycorestab/ExponentialFilter.order = {released_macro(config, 'stability/ExponentialFilter.order')}",
                          f"dycorestab/ExponentialFilter.tau = '{float(value) * substeps / s0} {unit}'"]
    elif stab:
        raise ValueError("stab needs damp")
    return "\n".join(overrides), hold, relax


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--variants", default="released;fft;hold2;hold3;fft+hold2")
    ap.add_argument("--days", type=float, default=10.0)
    ap.add_argument("--members", type=int, default=3, help="round-off perturbed members of the released model (0: paired test only)")
    ap.add_argument("--roundoff", type=float, default=1e-6,
                    help="relative input perturbation of the floor's members (1e-7 is below float32's epsilon)")
    ap.add_argument("--margin", type=float, default=0.01,
                    help="non-inferiority margin, relative to the released model's RMSE at each lead")
    ap.add_argument("--data-tag", default="jan2020")
    ap.add_argument("--tuned", type=Path, help="checkpoint whose parameters the ``tuned`` variants use")
    ap.add_argument("--fourier", choices=("fft", "matmul"), default=None,
                    help="fourier_method on every configuration (FastSphericalHarmonics checkpoints, the 0.7 degree "
                         "model, whose GPU default conflicts with its stacked transforms)")
    ap.add_argument("--resume", action="store_true", help="reuse the configurations a killed run already scored "
                                                            "(skill_<tag>.partial.json)")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    with open(args.model, "rb") as fh:
        ckpt = pickle.load(fh)
    base = f"RealSphericalHarmonicsWithZeroImag.fourier_method = '{args.fourier}'" if args.fourier else ""

    def with_base(overrides: str) -> str:
        return "\n".join(o for o in (base, overrides) if o)

    probe = SplitModel(ckpt, base)
    grid = probe.api.data_coords.horizontal
    label = f"{grid.longitude_nodes}x{grid.latitude_nodes}"
    init_ds = xarray.open_dataset(DATA / f"era5_{label}_{args.data_tag}_init.nc")
    verif = xarray.open_dataset(DATA / f"era5_{label}_{args.data_tag}_verif.nc")
    levels = list(np.asarray(probe.api.data_coords.vertical.centers))
    lat = np.deg2rad(np.asarray(verif.latitude))
    w_lat = jnp.asarray(np.cos(lat) / np.cos(lat).mean(), jnp.float32)
    steps_per_6h = round(np.timedelta64(6, "h") / probe.api.timestep)
    n_out = round(args.days * 4)
    n_inits = init_ds.sizes["time"]
    print(f"devices {jax.devices()} | {label} | model step {probe.api.timestep} | {probe.num_substeps} physics "
          f"intervals per step | {n_inits} initial times x {args.days} days | variants {args.variants}", flush=True)

    truth = {}
    for i in range(n_inits):
        t0 = init_ds.time.values[i]
        times = t0 + np.arange(1, n_out + 1) * np.timedelta64(6, "h")
        truth[i] = {f"{v}_{lev}": jnp.asarray(verif[f"{v}_{lev}"].sel(time=times).transpose("time", "longitude",
                                                                                              "latitude").values)
                    for v, lev in SCORED}

    def rmse(pred, ref):  # [lead, lon, lat]
        return jnp.sqrt(jnp.mean((pred - ref) ** 2 * w_lat, axis=(-2, -1)))

    def activity(pred):  # latitude-weighted spatial standard deviation: a smoothed forecast scores lower RMSE
        mean = jnp.mean(pred * w_lat, axis=(-2, -1), keepdims=True)
        return jnp.sqrt(jnp.mean((pred - mean) ** 2 * w_lat, axis=(-2, -1)))

    rows = {}
    rounds = [(name, None) for name in args.variants.split(";")]
    rounds += [("released", k) for k in range(args.members)]
    for name in args.variants.split(";"):  # every configuration builds before any forecast runs
        SplitModel(ckpt, with_base(parse_variant(name, ckpt["model_config_str"])[0]))
    RES.mkdir(parents=True, exist_ok=True)
    partial_out = RES / f"skill_{args.tag}.partial.json"
    if args.resume and partial_out.exists():
        # a run killed part-way (host OOM at 0.7 degree) keeps the per-start scores of every configuration it finished
        rows.update(json.loads(partial_out.read_text())["rows"])
        print(f"resumed: {sorted(rows)} from {partial_out.name}", flush=True)
    for name, member in rounds:
        if (name if member is None else f"released:roundoff{member}") in rows:
            continue
        overrides, hold, relax = parse_variant(name, ckpt["model_config_str"])
        m = SplitModel(ckpt, with_base(overrides))
        params = m.params
        if "tuned" in name.split("+"):
            with open(args.tuned, "rb") as fh:
                params = pickle.load(fh)["params"]
        n_int = m.num_substeps

        def forecast(x, forcings, m=m, params=params, hold=hold, n_int=n_int, relax=relax):
            """lax.scan over 6-hour blocks of model steps; physics refreshed every ``hold`` intervals.
            With ``relax``, a held interval's tendency is T_n + J (x - x_n), J fitted from the last
            two refreshes (``_relaxation_slope``); without, it is T_n (a frozen push)."""
            f0 = m.forcing_at(params, forcings, x)
            tend0 = m.physics(params, x, f0)
            s0 = _tendency_like(tend0, x.state)
            zero = jax.tree_util.tree_map(lambda t: jnp.zeros(t.shape[:1] + (1,) * (t.ndim - 1), t.dtype), tend0)

            def interval(carry, j):
                x, held, t_last, s_last, slope = carry
                f = m.forcing_at(params, forcings, x)
                s_now = _tendency_like(tend0, x.state)

                def refresh():
                    t_new = m.physics(params, x, f)
                    new_slope = _relaxation_slope(t_new, t_last, s_now, s_last) if relax else slope
                    return t_new, t_new, s_now, new_slope

                def keep():
                    if not relax:
                        return held, t_last, s_last, slope
                    lin = jax.tree_util.tree_map(lambda t, a, sn, so: t + a * (sn - so), t_last, slope, s_now, s_last)
                    return lin, t_last, s_last, slope

                held, t_last, s_last, slope = jax.lax.cond(j % hold == 0, refresh, keep)
                return (m.finish(params, x, held, f), held, t_last, s_last, slope), None

            def block(carry, b):
                ids = b * steps_per_6h * n_int + jnp.arange(steps_per_6h * n_int)
                carry = jax.lax.scan(interval, carry, ids)[0]
                out = m.api.decode(carry[0], forcings)
                return carry, {f"{v}_{lev}": out[v][levels.index(lev)] for v, lev in SCORED}

            return jax.lax.scan(block, (x, tend0, tend0, s0, zero), jnp.arange(n_out))[1]

        run = jax.jit(forecast)
        scores, ms = [], None
        for i in range(n_inits):
            ds0 = init_ds.isel(time=i)
            inputs = m.api.inputs_from_xarray(ds0)
            forcings = m.api.forcings_from_xarray(ds0)
            if member is not None:
                key = jax.random.key(100 + member)
                inputs = {k: (v * (1.0 + args.roundoff * jax.random.normal(jax.random.fold_in(key, j), np.shape(v)))
                              if np.issubdtype(np.asarray(v).dtype, np.floating) and k != "sim_time" else v)
                          for j, (k, v) in enumerate(inputs.items())}
            x0 = m.api.encode(inputs, forcings, jax.random.key(0))
            out = run(x0, forcings)
            scores.append({k: np.asarray(rmse(out[k], truth[i][k])) for k in out}
                          | {f"activity:{k}": np.asarray(activity(out[k])) for k in out})
            if i == 0 and member is None:
                windows = []
                for _ in range(2):
                    t0 = time.perf_counter()
                    jax.block_until_ready(run(x0, forcings))
                    windows.append(time.perf_counter() - t0)
                ms = 1e3 * min(windows) / (n_out * steps_per_6h)
        key_name = name if member is None else f"released:roundoff{member}"
        rows[key_name] = {"hold": hold, "ms_per_model_step": ms,
                          "rmse": {k: np.stack([s[k] for s in scores]).tolist() for k in scores[0]
                                   if not k.startswith("activity:")},
                          "activity": {k.split(":", 1)[1]: np.stack([s[k] for s in scores]).tolist()
                                       for k in scores[0] if k.startswith("activity:")}}
        partial_out.write_text(json.dumps({"model": str(args.model), "rows": rows}))  # survives a later failure
        mean = {k: np.stack([s[k] for s in scores]).mean(0) for k in scores[0]}
        print(f"{key_name:24s} " + (f"{ms:6.2f} ms/step | " if ms else "") + " | ".join(
            f"{k} day 1/3/5/10 " + "/".join(f"{mean[k][round(d * 4) - 1]:.1f}" for d in (1, 3, 5, 10) if round(d * 4) <= n_out)
            for k in mean), flush=True)

    # skill-equivalence against the round-off floor (none with --members 0), and the paired test
    ref = {k: np.asarray(rows["released"]["rmse"][k]) for k in rows["released"]["rmse"]}
    floor = {k: np.stack([np.asarray(rows[f"released:roundoff{j}"]["rmse"][k]) - ref[k]
                          for j in range(args.members)] or [np.zeros_like(ref[k])]) for k in ref}  # [member, init, lead]
    verdict = {}
    for name in (n for n in rows if not n.startswith("released")):
        verdict[name] = {}
        for k in ref:
            d = np.asarray(rows[name]["rmse"][k]) - ref[k]  # [init, lead]
            dm = d.mean(0)
            fm = floor[k].mean(1)  # per member, mean over inits
            band = (2.0 * np.maximum(fm.std(0, ddof=1) if args.members > 1 else np.abs(fm[0]), 1e-6) + np.abs(fm.mean(0))
                    if args.members else np.full_like(dm, np.nan))
            ok = (dm <= band)
            mean, half, n_eff = paired_interval(d)
            rel = ref[k].mean(0)
            cert = (mean + half) <= args.margin * rel
            verdict[name][k] = {"delta_rmse_mean": dm.tolist(), "roundoff_band": band.tolist(),
                                "equivalent_all_leads": bool(ok.all()), "first_failing_lead_h": None if ok.all()
                                else int(6 * (np.argmin(ok) + 1)),
                                "delta_rmse_rel": (mean / rel).tolist(), "ci95_rel": (half / rel).tolist(),
                                "n_eff": n_eff.tolist(), "margin": args.margin,
                                "certified": bool(cert.all()),
                                "first_uncertified_lead_h": None if cert.all() else int(6 * (np.argmin(cert) + 1))}
        print(f"{name:24s} " + " | ".join(
            f"{k}: dRMSE day 1/5/10 {verdict[name][k]['delta_rmse_mean'][3]:+.2f}/"
            f"{verdict[name][k]['delta_rmse_mean'][min(19, n_out - 1)]:+.2f}/{verdict[name][k]['delta_rmse_mean'][-1]:+.2f}"
            + (f" (round-off band day 10 {verdict[name][k]['roundoff_band'][-1]:.2f}) "
               f"{'equivalent' if verdict[name][k]['equivalent_all_leads'] else 'fails at ' + str(verdict[name][k]['first_failing_lead_h']) + ' h'}"
               if args.members else "")
            for k in ref), flush=True)
        print(f"{'':24s} paired, relative: " + " | ".join(
            f"{k} day 1/3/5/10 " + "/".join(
                f"{100 * verdict[name][k]['delta_rmse_rel'][min(round(day * 4), n_out) - 1]:+.1f}"
                f"+-{100 * verdict[name][k]['ci95_rel'][min(round(day * 4), n_out) - 1]:.1f}%" for day in (1, 3, 5, 10))
            + (f" certified at {100 * args.margin:.0f}%" if verdict[name][k]["certified"] else
               f" not certified from {verdict[name][k]['first_uncertified_lead_h']} h")
            for k in ref), flush=True)

    out = RES / f"skill_{args.tag}.json"
    out.write_text(json.dumps({"model": str(args.model), "days": args.days, "inits": [str(t) for t in init_ds.time.values],
                               "rows": rows, "verdict": verdict}, indent=1))
    partial_out.unlink(missing_ok=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
