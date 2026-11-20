"""Kochkov et al.'s Re 4000 Kolmogorov benchmark, on our classical solver and the planner (Part D, D5).

Their public Re 4000 evaluation set (``gs://gresearch/jax-cfd/public_eval_datasets/kolmogorov_re_4000``): 16
trajectories of a 4096^2 finite-volume DNS on a 4 pi box (viscosity 5e-4, forcing 0.5 sin(2y), drag 0.05), saved
on a 128^2 staggered grid every 0.0701 for 90 time units, plus their finite-volume baselines at 128-2048 and the
rollout of their learned interpolation (LI) at 128^2. Halving lengths and times maps that flow exactly onto ours:
Kolmogorov forcing sin(4y), drag 0.1, on [0, 2 pi) with viscosity 2.5e-4, velocities unchanged and vorticity
doubled. So their trajectories are read straight into ``frost.cfd.spectral`` (Grid(n, nu=2.5e-4)) with no new
physics.

Stages
- ``convert`` (``.venvs/frost-jaxcfd``, which reads netCDF): every file's staggered velocities (u at x-faces,
  v at y-faces, jax-cfd's offsets (1, 0.5) and (0.5, 1)) to spectral vorticity on our grid points, by the
  phase shifts of those offsets; frames up to ``--t-max`` (our time units); saved as npz;
- ``frontier`` (``.venvs/frost-mhd``): from each trajectory's first 128^2 state, our filtered pseudo-spectral
  solver at 128-384 (the classical frontier, timed at batch 16), converged dynamics at ``--n-dyn`` (the
  initial-condition ceiling), and their files as they are; all scored by the vorticity correlation against
  their 4096 DNS on the 128 band, tau = time until the sample-mean correlation drops below 0.95, reported in
  their time units.

Same-information protocol: every solver here starts from the 128^2 state, the only one the files contain.

Output: data/baselines/kochkov/re4000/vort_<name>.npz, data/results/e14/kochkov_re4000_frontier.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data" / "baselines" / "kochkov" / "re4000"
RES = ROOT / "data" / "results" / "e14"
FILES = {"dns4096": "eval_4096x4096_128x128.nc", "fv128": "eval_128x128_128x128.nc",
         "fv256": "eval_256x256_128x128.nc", "fv512": "eval_512x512_128x128.nc",
         "fv1024": "eval_1024x1024_128x128.nc", "fv2048": "eval_2048x2048_128x128.nc",
         "li128": "learned_interpolation_long_eval_128x128_128x128.nc"}
TIME_SCALE = 2.0  # their time = 2 x ours
FRAME_DT = 0.07012483601762931 / TIME_SCALE  # ours


def convert(args) -> int:
    import xarray

    n = 128
    k = np.fft.fftfreq(n) * n  # our wavenumbers on [0, 2 pi)
    kx, ky = k[:, None], np.fft.rfftfreq(n)[None, :] * n
    dx = 2 * np.pi / n
    # samples at (i + ox) dx, (j + oy) dx -> values at grid points i dx: multiply the FFT by exp(-i k . o dx)
    shift_u = np.exp(-1j * (kx * 1.0 + ky * 0.5) * dx)
    shift_v = np.exp(-1j * (kx * 0.5 + ky * 1.0) * dx)
    n_frames = int(args.t_max / FRAME_DT) + 1
    for name, fname in FILES.items():
        out = BASE / f"vort_{name}.npz"
        if out.exists():
            continue
        ds = xarray.open_dataset(BASE / fname)
        u = np.asarray(ds["u"].isel(time=slice(0, n_frames)).values, dtype=np.float64)  # (sample, time, x, y)
        v = np.asarray(ds["v"].isel(time=slice(0, n_frames)).values, dtype=np.float64)
        uh = np.fft.rfft2(u, axes=(-2, -1)) * shift_u
        vh = np.fft.rfft2(v, axes=(-2, -1)) * shift_v
        div = np.abs(1j * kx * uh + 1j * ky * vh).mean() / np.abs(1j * kx * vh - 1j * ky * uh).mean()
        w_hat = (1j * kx * vh - 1j * ky * uh)  # omega' = dv/dx' - du/dy' on our grid (= 2 omega theirs)
        np.savez(out, w_hat=w_hat.astype(np.complex64), t=np.arange(w_hat.shape[1]) * FRAME_DT)
        print(f"{name}: {w_hat.shape} -> {out.name} | |div| / |curl| = {div:.2e} (staggered offsets check)",
              flush=True)
    return 0


def frontier(args) -> int:
    import jax
    import jax.numpy as jnp

    from frost.cfd import spectral
    from frost.cfd.spectral import Grid

    nu = 2.5e-4
    g128 = Grid(128, nu=nu)
    truth = np.load(BASE / "vort_dns4096.npz")["w_hat"]  # (16, T, 128, 65)
    n_samples, n_t = truth.shape[:2]
    band = np.asarray(g128.mask, bool)

    def corr_curve(pred):  # pred, truth: (S, T, 128, 65) spectral on the 128 grid -> sample-mean correlation (T,)
        n = min(pred.shape[1], truth.shape[1])
        p = np.fft.irfft2(pred[:, :n] * band, s=(128, 128), axes=(-2, -1))
        q = np.fft.irfft2(truth[:, :n] * band, s=(128, 128), axes=(-2, -1))
        num = (p * q).sum((-2, -1))
        den = np.sqrt((p * p).sum((-2, -1)) * (q * q).sum((-2, -1)))
        corr = np.where(np.isfinite(num / den), num / den, 0.0)  # a non-finite frame has lost the flow
        return corr.mean(0)

    def tau(curve):
        below = np.nonzero(curve < 0.95)[0]
        t = np.arange(len(curve)) * FRAME_DT * TIME_SCALE
        return float(t[below[0]]) if len(below) else float(t[-1])

    out = {"protocol": "same information (128^2 initial state); tau in their time units", "rows": {}}
    for name in FILES:
        if name == "dns4096":
            continue
        pred = np.load(BASE / f"vort_{name}.npz")["w_hat"]
        out["rows"][name] = {"tau": tau(corr_curve(pred)), "source": "their file"}
        print(f"{name:8s} (theirs): tau {out['rows'][name]['tau']:.2f}", flush=True)

    w0 = jnp.asarray(truth[:, 0])
    steps_out = min(int(round(args.horizon / TIME_SCALE / FRAME_DT)), n_t - 1)
    for spec in args.solvers.split(","):
        n, dt_div = spec.split(":")
        n, dt = int(n), FRAME_DT / int(dt_div)
        g = Grid(n, nu=nu, dealias="exp")
        u0 = spectral.pad(g128, g, w0) if n > 128 else w0
        sub = int(dt_div)

        @jax.jit
        def roll(w):
            def frame(x, _):
                x = jax.lax.fori_loop(0, sub, lambda _, y: spectral.step(g, y, dt), x)
                return x, (spectral.truncate(g, g128, x) if n > 128 else x)
            return jax.lax.scan(frame, w, None, length=steps_out)[1]

        traj = jax.block_until_ready(roll(u0))  # compile + run
        t0 = time.perf_counter()
        traj = jax.block_until_ready(roll(u0))
        wall = time.perf_counter() - t0
        pred = np.concatenate([np.asarray(w0)[:, None], np.moveaxis(np.asarray(traj), 0, 1)], axis=1)
        finite = bool(np.isfinite(pred).all())
        tau_n = tau(corr_curve(pred)) if finite else 0.0
        ms_unit = 1e3 * wall / (steps_out * FRAME_DT * TIME_SCALE) / n_samples  # per their time unit, per trajectory
        key = f"spectral{n}:dt/{dt_div}"
        out["rows"][key] = {"tau": tau_n, "finite": finite, "ms_per_time_unit_per_traj": ms_unit, "n": n,
                            "dt_ours": dt, "batch": n_samples}
        print(f"{key:20s}: tau {tau_n:.2f} | {ms_unit:.3f} ms per time unit per trajectory (batch {n_samples})",
              flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"kochkov_re4000_frontier{args.tag}.json").write_text(json.dumps(out, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("convert", "frontier"))
    ap.add_argument("--t-max", type=float, default=15.0, help="convert: frames kept, our time units (= 30 theirs)")
    ap.add_argument("--horizon", type=float, default=30.0, help="frontier: their time units")
    ap.add_argument("--solvers", default="128:3,128:4,192:5,256:7,384:10,512:14,1024:40",
                    help="n:sub, our step = frame dt / sub (the last, the converged-dynamics ceiling)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    return {"convert": convert, "frontier": frontier}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
