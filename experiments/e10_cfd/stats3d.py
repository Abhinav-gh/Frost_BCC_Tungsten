"""Long-run statistics of W3 (forced isotropic turbulence, nu 3e-3, DNS 192^3): classical
solvers and the learned hybrids, at every method's own cost.

The forecast metric tau is capped by the initial condition: each method starts from the DNS
truncated to its own grid, so a coarse model competes against classical runs that know more
of the initial state (``ceiling3d.py``). What a large-eddy simulation is run for is its
statistics, which a long run computes after forgetting its initial state; there the
question is only how close each method's stationary statistics come to the DNS, per unit
cost (Kochkov et al. 2021, List et al. 2022, Novati et al. 2021 score LES this way).

Metrics, on the 32^3 band (shells 1..15), averaged over the 16 eval trajectories and over
t in [--t0, --horizon]:
- spectrum: mean |log(E(k) / E_DNS(k))| over shells (Kochkov's spectrum metric), also over
  the shells above ``--kh`` alone, where coarse grids go wrong;
- energy and band enstrophy (sum k^2 E): relative error of the means;
- skewness and flatness of the longitudinal velocity derivatives on the band: a spectrum can be
  matched shell by shell by a tuned eddy viscosity, the gradient statistics cannot.
The DNS reference is ``train/u32`` (8 trajectories x 20 time units after a 10-unit spin-up).
Its own floor is the same metric between its two halves (trajectories 0-3 against 4-7),
G3's rule: a difference smaller than the reference's half-to-half spread is not a difference.

Output: data/results/e10/stats3d_<tag>.json

Usage (guarded launcher; GPU, JAX env)
  scripts/safe_run.sh cfd_stats3d env -u LD_LIBRARY_PATH -u CUDA_HOME PYTHONPATH=src \\
      .venvs/frost-mhd/bin/python experiments/e10_cfd/stats3d.py --tag base
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from frost.cfd import pipeline3d, spectral3d as s3, split3d
from frost.cfd.spectral3d import Grid3

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w1_eval import RES, best_of  # noqa: E402
from w3split import load_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "cfd" / "hit192.h5"
MODELS = ROOT / "data" / "models" / "cfd"
DT_C = 0.04
KB = 15
AXES3 = (-3, -2, -1)
CLASSICAL = ("32:1:none:0,32:1:cl:1,32:1:smag:0.1,48:1:none:0,48:1:cl:0.5,48:1:cl:1,48:1:smag:0.1,"
             "56:1:none:0,64:1:none:0,64:1:cl:1,80:2:none:0,96:2:none:0")
LEARNED = "hit32_k1:1,hit32_k1:8,hit48epd_k1:1,hit48epd_k1:16,hit48epd_k16:16,hit48epd_k16:32"


def shell_spectrum(g32: Grid3, u_hat):
    """E(k) on shells 1..KB, summed over components, for each leading index: (..., KB)."""
    shell = np.rint(np.sqrt(g32.k2)).astype(np.int32)
    e = 0.5 * g32.weight * jnp.sum(jnp.abs(u_hat) ** 2, axis=-4)  # (..., n, n, n//2+1)
    flat = e.reshape(*e.shape[:-3], -1)
    onehot = jnp.asarray((shell.reshape(-1)[:, None] == np.arange(1, KB + 1)[None, :]).astype(np.float32))
    return flat @ onehot


GRAD_BAND = 12  # the shells a 32^3 grid holds outside its own exponential filter (trap 26)


def gradient_moments(g32: Grid3, u_hat, kmax: float | None = None):
    """Longitudinal velocity-derivative moments, averaged over the three directions: (<d^2>, <d^3>,
    <d^4>) per leading index, from which skewness and flatness follow; on |k| <= ``kmax`` if given."""
    ks = (g32.kx, g32.ky, g32.kz)
    if kmax is not None:
        u_hat = u_hat * jnp.asarray((np.sqrt(g32.k2) <= kmax + 0.5).astype(np.float32))
    out = []
    for i in range(3):
        d = jnp.fft.irfftn(1j * ks[i] * u_hat[..., i, :, :, :], s=(g32.n,) * 3, axes=AXES3)
        out.append(jnp.stack([jnp.mean(d**2, axis=AXES3), jnp.mean(d**3, axis=AXES3), jnp.mean(d**4, axis=AXES3)], -1))
    return sum(out) / 3.0


def band_stats(g32: Grid3, u_hat):
    """(KB shells of E(k); <d^2>, <d^3>, <d^4> on the full band; the same on |k| <= GRAD_BAND) per
    leading index. The second set is the like-for-like gradient statistic for a 32^3 model, whose
    filter damps the band's top shells, where gradients live, for every method on it. For an MHD state
    (six components) the same block for the magnetic field follows, then the cross helicity <u . b>."""
    if u_hat.shape[-4] == 6:
        u, b = u_hat[..., :3, :, :, :], u_hat[..., 3:, :, :, :]
        hc = jnp.sum(jnp.real(jnp.conj(u) * b) * g32.weight, axis=(-4, -3, -2, -1)) / float(g32.n) ** 6
        return jnp.concatenate([band_stats(g32, u), band_stats(g32, b), hc[..., None]], axis=-1)
    return jnp.concatenate([shell_spectrum(g32, u_hat), gradient_moments(g32, u_hat),
                            gradient_moments(g32, u_hat, GRAD_BAND)], axis=-1)


BLOCK = KB + 6  # one field's statistics in band_stats


def scores_any(spec, ref, kh: int, shells: int = KB) -> dict:
    """``scores`` for a flow, or for MHD the kinetic scores, the magnetic ones (prefixed ``mag_``) and the
    cross-helicity error relative to the DNS's total energy."""
    if len(spec) < 2 * BLOCK:
        return scores(spec, ref, kh, shells)
    out = scores(spec[:BLOCK], ref[:BLOCK], kh, shells)
    out |= {f"mag_{k}": v for k, v in scores(spec[BLOCK:2 * BLOCK], ref[BLOCK:2 * BLOCK], kh, shells).items()}
    e_tot = ref[:shells].sum() + ref[BLOCK:BLOCK + shells].sum()
    out["cross_helicity_err"] = float((spec[-1] - ref[-1]) / max(e_tot, 1e-30))
    return out


def scores(spec, ref, kh: int, shells: int = KB) -> dict:
    """spec, ref: (KB + 3,) time- and trajectory-averaged shell spectra then gradient moments; the
    spectrum is scored on shells 1..``shells``."""
    k = np.arange(1, shells + 1)
    m, mr = spec[KB:KB + 3], ref[KB:KB + 3]
    out = {}
    if len(spec) >= KB + 6 and len(ref) >= KB + 6:  # gradient moments on |k| <= GRAD_BAND as well
        mb, mbr = spec[KB + 3:KB + 6], ref[KB + 3:KB + 6]
        out = {"skewness_band": float(mb[1] / mb[0] ** 1.5),
               "skewness_band_err": float(mb[1] / mb[0] ** 1.5 - mbr[1] / mbr[0] ** 1.5),
               "flatness_band": float(mb[2] / mb[0] ** 2),
               "flatness_band_err": float(mb[2] / mb[0] ** 2 - mbr[2] / mbr[0] ** 2)}
    spec, ref = spec[:shells], ref[:shells]
    lr = np.abs(np.log(spec / ref))
    skew, skew_ref = m[1] / m[0] ** 1.5, mr[1] / mr[0] ** 1.5
    flat, flat_ref = m[2] / m[0] ** 2, mr[2] / mr[0] ** 2
    return {"spectrum_l1": float(lr.mean()), "spectrum_l1_high": float(lr[k > kh].mean()),
            "energy_err": float(spec.sum() / ref.sum() - 1.0),
            "enstrophy_err": float((k**2 * spec).sum() / (k**2 * ref).sum() - 1.0),
            "skewness": float(skew), "skewness_err": float(skew - skew_ref),
            "flatness": float(flat), "flatness_err": float(flat - flat_ref),
            **out, "spectrum_ratio": (spec / ref).tolist()}


@partial(jax.jit, static_argnames=("grid", "score", "kind", "c", "m", "n_records", "every", "physics"))
def classical_rollout(grid: Grid3, score: Grid3, u0, *, kind: str, c: float, m: int, n_records: int, every: int,
                      physics: str = "hydro"):
    def record(u, _):
        u = jax.lax.fori_loop(0, every, lambda _, x: split3d.physics_step(grid, x, DT_C, (kind, c), physics, m), u)
        return u, band_stats(score, u if grid == score else s3.truncate(grid, score, u))

    return jax.lax.scan(record, u0, None, length=n_records)[1]


@partial(jax.jit, static_argnames=("grid", "kind", "c", "m", "n_steps", "physics"))
def classical_advance(grid: Grid3, u0, *, kind: str, c: float, m: int, n_steps: int, physics: str = "hydro"):
    return jax.lax.fori_loop(0, n_steps, lambda _, x: split3d.physics_step(grid, x, DT_C, (kind, c), physics, m), u0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--classical", default=CLASSICAL,
                    help="n:substeps per 0.04:closure:c[:t],...; t scales the filter per unit time (6/m per step)")
    ap.add_argument("--learned", default=LEARNED,
                    help="model:hold k[:p|:s],... (data/models/cfd); p: the closure on the second device one hold "
                         "behind (frost.cfd.pipeline3d), s: the same loop serialised")
    ap.add_argument("--horizon", type=float, default=40.0)
    ap.add_argument("--t0", type=float, default=4.0)
    ap.add_argument("--every", type=int, default=10, help="steps of 0.04 between records")
    ap.add_argument("--kh", type=int, default=8, help="shells above this are scored separately")
    ap.add_argument("--shells", type=int, default=KB,
                    help="spectrum scored on shells 1..this: a 32^3 grid's own exponential filter damps shells "
                         "14-15 to a fifth for every method on it, so a 32^3 LES is judged on 1..12")
    ap.add_argument("--batch", type=int, default=16, help="timing batch")
    ap.add_argument("--steps", type=int, default=32, help="timed length, in steps of 0.04")
    ap.add_argument("--timed-steps", type=int, default=256,
                    help="timed length of a pipelined run (whole hold blocks), in steps of 0.04")
    ap.add_argument("--data", type=Path, default=DATA, help="DNS file: train/u32 (reference), eval/u_ic (starts)")
    ap.add_argument("--ref-from", type=float, default=0.0,
                    help="fraction of the DNS records skipped as transient: the reference must be stationary "
                         "(hit288 after a 6-unit spin-up still gains 25%% energy over its first 6 units)")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    with h5py.File(args.data, "r") as fh:
        attrs = dict(fh.attrs)
        train = np.asarray(fh["train/u32"])  # (T, B, 3, 32, 32, 32)
        ev_ic = np.asarray(fh["eval/u_ic"])
    nu, power = float(attrs["nu"]), float(attrs["power"])
    power_b = float(attrs.get("power_b", -1.0))  # MHD: power into the field (< 0: as the flow)
    physics = str(attrs.get("physics", "hydro"))
    g32 = Grid3(32, nu=nu, power=power, power_b=power_b, dealias="exp")
    g_ic = Grid3(int(attrs["n_ic"]), nu=nu, power=power, power_b=power_b, dealias="exp")
    u_ic = s3.to_hat(g_ic, jnp.asarray(ev_ic))
    del ev_ic

    # the DNS reference and its own half-to-half floor
    dns_fn = jax.jit(lambda u: band_stats(g32, s3.to_hat(g32, u)))
    first = round(args.ref_from * train.shape[0])
    spec_dns = np.stack([np.asarray(dns_fn(jnp.asarray(train[t]))) for t in range(first, train.shape[0])])
    with h5py.File(args.data, "r") as fh:
        st = np.asarray(fh["train/stats"])[first:]
        energy = st[:, 0] + (st[:, 1] if physics == "mhd" else 0.0)  # MHD: kinetic plus magnetic
    thirds = [float(x.mean()) for x in np.array_split(energy, 3)]
    print(f"DNS reference: records {first}..{train.shape[0]}, energy by thirds " + " / ".join(f"{x:.3f}" for x in thirds)
          + (" (drifting: not stationary)" if abs(thirds[2] / thirds[0] - 1.0) > 0.03 else " (stationary)"), flush=True)
    half = train.shape[1] // 2
    ref = spec_dns.mean(axis=(0, 1))
    floor = scores_any(spec_dns[:, :half].mean(axis=(0, 1)), spec_dns[:, half:].mean(axis=(0, 1)), args.kh, args.shells)
    del train
    n_rec = round(args.horizon / (DT_C * args.every))
    keep = slice(round(args.t0 / (DT_C * args.every)), None)
    print(f"devices {jax.devices()} | DNS {spec_dns.shape[0]} records x {spec_dns.shape[1]} trajectories | "
          f"floor (half against half): spectrum {floor['spectrum_l1']:.4f}, high shells "
          f"{floor['spectrum_l1_high']:.4f}, energy {floor['energy_err']:+.4f}, enstrophy {floor['enstrophy_err']:+.4f}, "
          f"skewness {floor['skewness_err']:+.4f} (DNS {floor['skewness']:.3f}), flatness {floor['flatness_err']:+.3f}, "
          f"band<={GRAD_BAND} skewness {floor['skewness_band_err']:+.4f} (DNS {floor['skewness_band']:.3f}) "
          f"flatness {floor['flatness_band_err']:+.3f} (DNS {floor['flatness_band']:.2f})",
          flush=True)

    rows = {}

    def report(name, spec_t, ms, extra, keep=keep):
        finite = bool(np.isfinite(spec_t).all())
        row = {"ms_per_0.04": ms, "finite": finite, **extra}
        if finite:
            mean_stats = spec_t[keep].mean(axis=(0, 1))
            row.update(scores_any(mean_stats, ref, args.kh, args.shells))
            row["mean_stats"] = mean_stats.tolist()  # shells then moments: re-scorable against another reference
            # per trajectory (time-averaged), so a pooled scorer can bootstrap over trajectories (pooled_stats.py)
            row["traj_stats"] = spec_t[keep].mean(axis=0).tolist()
        rows[name] = row
        RES.mkdir(parents=True, exist_ok=True)
        (RES / f"stats3d_{args.tag}.json").write_text(json.dumps(
            {"horizon": args.horizon, "t0": args.t0, "kh": args.kh, "floor": floor, "data": str(args.data),
             "ref_from": args.ref_from, "shells": args.shells, "dns_stats": ref.tolist(),
             "dns_traj_stats": spec_dns.mean(axis=0).tolist(), "dns_records": int(spec_dns.shape[0]), "rows": rows},
            indent=1))  # after every row: survives a later failure
        print(f"{name:24s} {ms:8.2f} ms | " + (
            f"spectrum {row['spectrum_l1']:.4f} (high {row['spectrum_l1_high']:.4f}) | energy {row['energy_err']:+.4f}"
            f" | enstrophy {row['enstrophy_err']:+.4f} | skew {row['skewness']:.3f} ({row['skewness_err']:+.3f})"
            f" | flat {row['flatness']:.2f} ({row['flatness_err']:+.2f})"
            + (f" | band<={GRAD_BAND}: skew {row['skewness_band']:.3f} ({row['skewness_band_err']:+.3f})"
               f" flat {row['flatness_band']:.2f} ({row['flatness_band_err']:+.2f})" if "skewness_band" in row else "")
            + (f" || MAG spectrum {row['mag_spectrum_l1']:.4f} | current {row['mag_enstrophy_err']:+.4f}"
               f" | band<={GRAD_BAND}: skew {row['mag_skewness_band']:.3f} ({row['mag_skewness_band_err']:+.3f})"
               f" flat {row['mag_flatness_band']:.2f} ({row['mag_flatness_band_err']:+.2f})"
               f" | cross helicity {row['cross_helicity_err']:+.4f}" if "mag_spectrum_l1" in row else "")
            if finite else "not finite"), flush=True)

    for run in (r for r in args.classical.split(",") if r):
        n, m, kind, c, *opt = run.split(":")
        n, m, c = int(n), int(m), float(c)
        # "t": the exp filter's per-step strength set to 6/m, so m substeps damp per unit time as one
        # 0.04 step does (the rate-coupled filter of trap 25, as `damp` fixes it for NeuralGCM)
        g = Grid3(n, nu=nu, power=power, power_b=power_b, dealias="exp", filter_strength=6.0 / m if "t" in opt else 6.0)
        u0 = s3.truncate(g_ic, g, u_ic)
        spec_t = np.asarray(classical_rollout(g, g32, u0, kind=kind, c=c, m=m, n_records=n_rec, every=args.every,
                                              physics=physics))
        ub = jnp.tile(u0, (-(-args.batch // u0.shape[0]), 1, 1, 1, 1))[:args.batch]
        lo, _ = best_of(lambda ub=ub, g=g, kind=kind, c=c, m=m: classical_advance(g, ub, kind=kind, c=c, m=m,
                                                                                  n_steps=args.steps, physics=physics))
        report(f"classical {run}", spec_t, 1e3 * lo / args.steps, {"n": n, "m": m, "closure": kind, "c": c})

    devices = jax.devices()
    for run in (r for r in args.learned.split(",") if r):
        name, k, *mode = run.split(":")
        k = int(k)
        if not (MODELS / f"{name}.json").exists():
            print(f"learned {run}: no model at {MODELS / name}, skipped", flush=True)
            continue
        model = load_model(MODELS / name)
        if "h" in mode:  # the network in bfloat16 (split3d.Split3Static.net_dtype)
            import dataclasses

            model["static"] = dataclasses.replace(model["static"], net_dtype="bf16")
            mode = [m for m in mode if m != "h"]
        hold_mode = "sync"
        if "x" in mode:  # the correction extrapolated across each hold from the last two refreshes
            hold_mode = "extrap"
            mode = [m for m in mode if m != "x"]
        meta = json.loads((MODELS / f"{name}.json").read_text())
        g = Grid3(int(meta["grid"]), nu=nu, power=power, power_b=power_b, dealias=meta.get("dealias", "exp"))
        u0 = s3.truncate(g_ic, g, u_ic)
        if mode:  # through frost.cfd.pipeline3d: "p" the closure on device 1 one hold behind, "s" serialised
            if len(devices) < 2:
                print(f"learned {run}: needs two devices, skipped", flush=True)
                continue
            lag = 1 if mode[0] == "p" else 0
            extrap = "e" in mode[1:]  # p:e, the lagged correction extrapolated one hold forward
            obs = jax.jit(lambda u, g=g: band_stats(g32, s3.truncate(g, g32, u)))
            n_blocks = round(args.horizon / (DT_C * k))
            _, recs = pipeline3d.run(g, model, u0, k=k, dt=DT_C, n_outer=n_blocks, lag=lag,
                                     solver_device=devices[0], closure_device=devices[1], observe=obs,
                                     extrapolate=extrap)
            spec_t = np.stack([np.asarray(r) for r in recs])
            keep_p = slice(round(args.t0 / (DT_C * k)), None)
            ub = jnp.tile(u0, (-(-args.batch // u0.shape[0]), 1, 1, 1, 1))[:args.batch]
            blocks = max(2, round(args.timed_steps / k))

            def timed(ub=ub, g=g, model=model, k=k, lag=lag, blocks=blocks, extrap=extrap):
                u, _ = pipeline3d.run(g, model, ub, k=k, dt=DT_C, n_outer=blocks, lag=lag,
                                      solver_device=devices[0], closure_device=devices[1], extrapolate=extrap)
                return u  # the loop's end after a device sync, before any teardown (trap 23)

            lo, _ = best_of(timed)
            report(f"learned {run}", spec_t, 1e3 * lo / (blocks * k),
                   {"model": name, "k": k, "n": g.n, "pipelined_lag": lag}, keep=keep_p)
            continue
        states = split3d.rollout(g, model, u0, k=k, dt=DT_C, n_records=n_rec, record_every=args.every,
                                 mode=hold_mode)
        spec_t = np.asarray(jax.vmap(lambda u, g=g: band_stats(g32, s3.truncate(g, g32, u)))(states))
        del states
        ub = jnp.tile(u0, (-(-args.batch // u0.shape[0]), 1, 1, 1, 1))[:args.batch]
        lo, _ = best_of(lambda ub=ub, g=g, model=model, k=k, hm=hold_mode: split3d.advance(
            g, model, ub, k=k, dt=DT_C, n_steps=args.steps, mode=hm))
        report(f"learned {run}", spec_t, 1e3 * lo / args.steps, {"model": name, "k": k, "n": g.n})

    RES.mkdir(parents=True, exist_ok=True)
    out = RES / f"stats3d_{args.tag}.json"
    out.write_text(json.dumps({"horizon": args.horizon, "t0": args.t0, "kh": args.kh, "floor": floor,
                               "data": str(args.data), "ref_from": args.ref_from, "dns_stats": ref.tolist(),
                               "dns_traj_stats": spec_dns.mean(axis=0).tolist(), "dns_records": int(spec_dns.shape[0]),
                               "rows": rows}, indent=1))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
