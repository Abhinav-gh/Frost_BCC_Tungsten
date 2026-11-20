"""The tracer gate for W1-T: is there a window for a learned scheme before anything is trained?

Every arm advects the same tracers through the same velocity: the W1 eval states
(Kolmogorov flow, Re 1000) run by filtered spectral on 64^2 at dt 0.014, the velocity
held over each 0.014 interval and padded onto each tracer grid. The tracers start
smooth, non-negative and band-limited (|k| <= 8, exact on every grid), so the
initial-condition ceiling that decides W1's velocity race is absent: what separates
the arms is the scheme alone.

Arms:
- **reference**: spectral 1024^2, 2/3 rule, diffusion kappa;
- **classical frontier**: spectral (exp filter), spectral with hole filling (positive),
  and the finite-volume schemes of ``frost.cfd.tracers`` (van Leer, WENO-Z with and
  without its positivity limiter) across resolution, each at its own CFL;
- **Markov bound** on n^2: the reference dynamics, projected onto the n^2 band after
  every 0.014 interval. It is what a perfect deterministic scheme holding only an n^2
  state reaches (zero completion of the unresolved band, as ``ceiling.py``), the bar
  for a learned scheme on that grid.

Scored on the 64^2 bands (2/3: |k| <= 21; full: <= 31) against the reference: tau
(first correlation dip below 0.95), relative L2 of the band at fixed times, the field
minimum (positivity) and the mass drift. Timing per 0.014 at 64 tracer fields.

Output: data/results/e10/tracer_frontier_<tag>.json

Usage (guarded launcher; GPU, JAX env)
  scripts/safe_run.sh cfd_tracer_gate env -u LD_LIBRARY_PATH -u CUDA_HOME PYTHONPATH=src \\
      .venvs/frost-mhd/bin/python experiments/e10_cfd/tracer_frontier.py --tag gate
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from functools import partial
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from frost.cfd import spectral, tracers
from frost.cfd.spectral import Grid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_closure import DATA  # noqa: E402
from w1_eval import RES, best_of, correlation_times  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

DT_V = 0.014  # flow step and velocity hold interval
REC_I = 8  # intervals per record (0.112 time units)
CFL_SPEC = 1.5  # IFRK4: |lambda| dt on the imaginary axis (limit 2.83), with 2D headroom
CFL_FV = 0.3  # u_max dt / dx per direction (van Leer is positive below 1/4 summed)
G64E = Grid(64, dealias="exp")  # the flow grid
TIMED_FIELDS = 64


def substeps(arm: str, grid: Grid, umax: float) -> int:
    if arm in ("spectral", "spectral_fill"):
        kmax = grid.n / 3 if grid.dealias == "two_thirds" else grid.n / 2
        return math.ceil(DT_V * kmax * umax / CFL_SPEC)
    return math.ceil(DT_V * umax / (CFL_FV * grid.dx))


def make_interval(arm: str, grid: Grid, sub: int, kappa: float, proj_kc: int | None):
    """One 0.014 interval of tracer transport under the held velocity ``w64``."""
    dt = DT_V / sub
    keep = None if proj_kc is None else tracers.band_mask(grid, proj_kc)

    def interval(c, w64):
        if arm in ("spectral", "spectral_fill"):
            u, v = tracers.point_velocity(G64E, grid, w64)
            u, v = u[:, None], v[:, None]
            c = jax.lax.fori_loop(0, sub, lambda _, x: tracers.spectral_step(grid, x, u, v, dt, kappa), c)
            if arm == "spectral_fill":
                c = tracers.hole_fill(grid, c)
            return c if keep is None else c * keep
        ux, vy = tracers.face_velocity(G64E, grid, w64)
        ux, vy = ux[:, None], vy[:, None]
        return jax.lax.fori_loop(0, sub, lambda _, x: tracers.fv_step(arm, grid.dx, x, ux, vy, dt, kappa), c)

    return interval


@partial(jax.jit, static_argnames=("arm", "grid", "sub", "kappa", "proj_kc"))
def rollout(c0, w_hist, *, arm: str, grid: Grid, sub: int, kappa: float, proj_kc: int | None):
    """Records of the 64^2 point-value spectrum, the field minimum and the mean, every REC_I intervals."""
    interval = make_interval(arm, grid, sub, kappa, proj_kc)
    ca = tracers.CellAverage(grid)
    n_rec = w_hist.shape[0] // REC_I
    w_rec = w_hist[: n_rec * REC_I].reshape(n_rec, REC_I, *w_hist.shape[1:])

    def record(c, w_k):
        c = jax.lax.scan(lambda x, w: (interval(x, w), None), c, w_k)[0]
        spec = arm in ("spectral", "spectral_fill")
        pts = c if spec else ca.to_points(c)
        raw = spectral.to_real(grid, c) if spec else c
        return c, (tracers.truncate_sharp(grid, 64, 31, pts), raw.min(axis=(-2, -1)), raw.mean(axis=(-2, -1)))

    return jax.lax.scan(record, c0, w_rec)[1]


@partial(jax.jit, static_argnames=("arm", "grid", "sub", "kappa"))
def advance(c0, w_hist, *, arm: str, grid: Grid, sub: int, kappa: float):
    interval = make_interval(arm, grid, sub, kappa, None)
    return jax.lax.scan(lambda x, w: (interval(x, w), None), c0, w_hist)[0]


def initial(arm: str, grid: Grid, c0_hat64):
    c_hat = c0_hat64 if grid.n == 64 else spectral.pad(G64E, grid, c0_hat64)
    return c_hat if arm in ("spectral", "spectral_fill") else tracers.CellAverage(grid).from_points(c_hat)


BANDS = {"k21": tracers.band_mask(G64E, 21), "k31": tracers.band_mask(G64E, 31)}


def band(name: str, spec64):
    """Real 64^2 field of a record's band (|kx|, |ky| <= 21 or 31; records are unfiltered)."""
    return spectral.to_real(G64E, spec64 * BANDS[name])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spectral", default="64,96,128,192,256,384,512", help="exp-filtered spectral sizes")
    ap.add_argument("--fill", default="", help="spectral sizes with hole filling every interval (positive)")
    ap.add_argument("--fv", default="64,96,128,192,256,384,512", help="finite-volume sizes")
    ap.add_argument("--schemes", default="vanleer,weno5z,weno5z_pp")
    ap.add_argument("--markov", default="64,128", help="Markov-bound projections (n^2 bands)")
    ap.add_argument("--ref", type=int, default=1024)
    ap.add_argument("--kappa", type=float, default=1e-4)
    ap.add_argument("--horizon", type=float, default=10.08, help="time units (a multiple of 0.112)")
    ap.add_argument("--traj", type=int, default=8)
    ap.add_argument("--tracers", type=int, default=2, help="tracers per trajectory")
    ap.add_argument("--timed-intervals", type=int, default=64)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    g128 = Grid(128)
    with h5py.File(args.data, "r") as fh:
        w128 = np.asarray(fh["eval/w128"][0, : args.traj])
    w0 = spectral.truncate(g128, G64E, jnp.asarray(np.fft.rfft2(w128) * g128.mask, jnp.complex64))
    n_int = round(args.horizon / DT_V)
    n_int -= n_int % REC_I

    @jax.jit
    def history(w):
        return jax.lax.scan(lambda x, _: (spectral.step(G64E, x, DT_V), x), w, None, length=n_int)[1]

    w_hist = history(w0)
    umax = float(jnp.max(jax.vmap(lambda w: spectral.max_speed(G64E, w))(w_hist)))
    c0 = tracers.random_tracer(G64E, jax.random.PRNGKey(7), batch=(args.traj, args.tracers))
    t_rec = REC_I * DT_V
    print(f"devices {jax.devices()} | {args.traj} flows x {args.tracers} tracers | {n_int} intervals of {DT_V} "
          f"| u_max {umax:.2f} | kappa {args.kappa:g}", flush=True)

    w_time = jnp.tile(w_hist[: args.timed_intervals], (1, 1, 1, 1))
    reps = -(-TIMED_FIELDS // (args.traj * args.tracers))

    def run(name, arm, grid, proj_kc=None, timed=True):
        sub = substeps(arm, grid, umax)
        x0 = initial(arm, grid, c0)
        spec, cmin, mean = rollout(x0, w_hist, arm=arm, grid=grid, sub=sub, kappa=args.kappa, proj_kc=proj_kc)
        row = {"arm": arm, "n": grid.n, "dealias": grid.dealias, "sub": sub, "dt": DT_V / sub,
               "min": float(jnp.min(cmin)), "neg_records": float(jnp.mean(cmin < 0.0)),
               "mass_drift": float(jnp.max(jnp.abs(mean - 1.0)))}  # the tracers start at mean 1
        if timed:
            xb = jnp.tile(x0, (1, reps, 1, 1))
            lo, hi = best_of(lambda: advance(xb, w_time, arm=arm, grid=grid, sub=sub, kappa=args.kappa))
            row["ms_per_0.014"] = 1e3 * lo / args.timed_intervals
            row["fields_timed"] = int(xb.shape[0] * xb.shape[1])
            row["window_ratio"] = lo / hi
        return row, spec

    ref_grid = Grid(args.ref)
    cache = ROOT / "data" / "cfd" / (f"tracer_ref{args.ref}_k{args.kappa:g}_T{n_int}_{args.traj}x{args.tracers}.npz")
    if cache.exists():
        z = np.load(cache)
        ref_row, ref_spec = json.loads(str(z["row"])), jnp.asarray(z["spec"])
        print(f"reference from {cache.name}", flush=True)
    else:
        ref_row, ref_spec = run(f"reference{args.ref}", "spectral", ref_grid, timed=False)
        np.savez(cache, spec=np.asarray(ref_spec), row=json.dumps(ref_row))
    ref = {b: band(b, ref_spec) for b in BANDS}
    rows = {f"reference{args.ref}": ref_row}

    def score(name, row, spec):
        for b in BANDS:
            a = band(b, spec)
            rho = np.asarray(spectral.correlation(a, ref[b])).reshape(a.shape[0], -1)
            tau = correlation_times(np.where(np.isfinite(rho), rho, -1.0), t_rec)
            err = np.asarray(jnp.sqrt(((a - ref[b]) ** 2).sum((-2, -1)) / ((ref[b] - ref[b].mean((-2, -1), keepdims=True)) ** 2).sum((-2, -1))))
            err = err.reshape(err.shape[0], -1).mean(-1)
            row[f"tau_{b}"] = float(tau.mean())
            row[f"tau_{b}_se"] = float(tau.std(ddof=1) / np.sqrt(tau.size))
            row[f"relL2_{b}"] = {f"{t:g}": float(err[round(t / t_rec) - 1]) for t in (1.008, 2.016, 5.04, 10.08)
                                 if round(t / t_rec) - 1 < err.size}
        rows[name] = row
        ms = row.get("ms_per_0.014")
        print(f"{name:22s} sub {row['sub']:3d} | tau k21 {row['tau_k21']:5.2f} +- {row['tau_k21_se']:.2f} | "
              f"tau k31 {row['tau_k31']:5.2f} | relL2 k31 {row['relL2_k31']} | min {row['min']:+.3f} | "
              f"mass {row['mass_drift']:.1e} | " + (f"{ms:.3f} ms per 0.014 ({row['fields_timed']} fields)" if ms else ""),
              flush=True)

    for n in (int(x) for x in args.markov.split(",") if x):
        row, spec = run(f"markov{n}", "spectral", ref_grid, proj_kc=n // 2 - 1, timed=False)
        score(f"markov{n}", row, spec)
    for n in (int(x) for x in args.spectral.split(",") if x):
        row, spec = run(f"spectral{n}", "spectral", Grid(n, dealias="exp"))
        score(f"spectral{n}", row, spec)
    for n in (int(x) for x in args.fill.split(",") if x):
        row, spec = run(f"spectral_fill{n}", "spectral_fill", Grid(n, dealias="exp"))
        score(f"spectral_fill{n}", row, spec)
    for scheme in (s for s in args.schemes.split(",") if s):
        for n in (int(x) for x in args.fv.split(",") if x):
            row, spec = run(f"{scheme}{n}", scheme, Grid(n, dealias="exp"))
            score(f"{scheme}{n}", row, spec)

    RES.mkdir(parents=True, exist_ok=True)
    out = RES / f"tracer_frontier_{args.tag}.json"
    out.write_text(json.dumps({"kappa": args.kappa, "umax": umax, "horizon": n_int * DT_V, "rows": rows,
                               "reference_cache": cache.name}, indent=1))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
