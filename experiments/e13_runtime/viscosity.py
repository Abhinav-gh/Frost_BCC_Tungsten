"""Shear viscosity by Green-Kubo from every arm's saved frames, with the teacher's own stress (2026-11-17).

further_expansions_beyond.md 6.7 (transport): conductivity needs only positions (conductivity.py); viscosity needs the
stress. Each arm's frames (``md_runtime.py --frames --frames-x-fs 5 --frames-v-fs 5``) are re-evaluated with the teacher
(graphed, edge forces, the virial tensor W_ab = -sum_e r_a dE/dr_b, the skin list filtered to r_max), so every arm is
scored by one stress estimator and only its dynamics differ. The pressure tensor is P_ab = (sum_i m v_a v_b + W_ab) / V
(eV/A^3; velocities in ASE units, m v^2 in eV), and

    eta = V / (k_B T) * int_0^tc < P_s(0) P_s(t) > dt

averaged over the five independent traceless components (P_xy, P_xz, P_yz, (P_xx - P_yy) / 2, (P_yy - P_zz) / 2), with T
the arm's own mean kinetic temperature. The running integral is read as its mean over [tc_lo, tc_hi] (its plateau), and
the error is the standard error over equal blocks of the run. eta in mPa s (1 eV fs / A^3 = 0.16022 mPa s).

    python experiments/e13_runtime/viscosity.py data/results/e13/frames_<run>_<tag>.npz [--blocks 5]
Output: data/results/e13/viscosity_<run>_<tag>.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "results" / "e13"
EV_FS_A3_TO_MPAS = 0.16021766
KB = 8.617333262e-5  # eV / K


def stress_series(calc, numbers, cell, xs, vs, masses, dev: str) -> np.ndarray:
    """(n, 3, 3) pressure tensors, eV/A^3, from positions xs and velocities vs (same frames) with the teacher."""
    from frost.neighbours import VerletList, tight_edges
    from frost.teacher import frame_atoms
    from frost.teacher_fast import GraphedTeacher

    atoms = frame_atoms(numbers, xs[0], cell)
    g = GraphedTeacher(calc, atoms, dev, edge_forces=True, virial=True)
    r2 = float(g.r_max) ** 2
    lst = VerletList(cell, g.r_max, 1.0, dev, wrap_positions=False, builder="alchemi")
    m = torch.as_tensor(masses, dtype=torch.float64, device=dev).view(-1, 1)
    vol = abs(float(np.linalg.det(cell)))
    out = np.zeros((len(xs), 3, 3))
    for f in range(len(xs)):
        x = torch.as_tensor(xs[f], dtype=torch.float32, device=dev)
        lst.update(x)
        g.set_edges(*tight_edges(lst, x, r2))
        g(x)
        w = g.static_virial_tensor.detach().double()
        v = torch.as_tensor(vs[f], dtype=torch.float64, device=dev)
        k = (m * v).T @ v
        out[f] = ((k + w) / vol).cpu().numpy()
    return out


def shear_components(p: np.ndarray) -> np.ndarray:
    """(n, 5): the traceless shear components of each pressure tensor."""
    return np.stack([p[:, 0, 1], p[:, 0, 2], p[:, 1, 2], 0.5 * (p[:, 0, 0] - p[:, 1, 1]),
                     0.5 * (p[:, 1, 1] - p[:, 2, 2])], axis=1)


def acf(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Mean autocorrelation over columns, lags 0..max_lag-1 (FFT, unbiased by the number of pairs per lag)."""
    n = series.shape[0]
    s = series - series.mean(axis=0)
    size = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(s, size, axis=0)
    c = np.fft.irfft(f * np.conj(f), size, axis=0)[:max_lag]
    c /= (n - np.arange(max_lag))[:, None]
    return c.mean(axis=1)


def green_kubo(comps: np.ndarray, dt_fs: float, vol: float, temp: float, tc_fs: float):
    """Running integral eta(t) (mPa s) up to tc_fs."""
    lags = max(2, int(round(tc_fs / dt_fs)) + 1)
    c = acf(comps, lags)
    run = np.concatenate([[0.0], np.cumsum(0.5 * (c[1:] + c[:-1]) * dt_fs)])
    return run * vol / (KB * temp) * EV_FS_A3_TO_MPAS, c


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames")
    ap.add_argument("--blocks", type=int, default=5, help="equal blocks for the standard error")
    ap.add_argument("--tc-lo", type=float, default=1000.0, help="fs: start of the plateau window")
    ap.add_argument("--tc-hi", type=float, default=2000.0, help="fs: end of the plateau window")
    ap.add_argument("--skip-fs", type=float, default=0.0, help="fs of each arm discarded first")
    args = ap.parse_args()
    from frost.teachers import load_calc, teacher_of

    path = Path(args.frames)
    d = np.load(path)
    stem = path.stem.removeprefix("frames_")
    run = re.match(r"(.*?_T\d+)(_|$)", stem).group(1)
    with __import__("h5py").File(ROOT / "data" / "trajectories" / f"{run}.h5", "r") as fh:
        teacher_name = teacher_of(fh.attrs)
    numbers, cell, masses = d["numbers"], np.asarray(d["cell"], dtype=np.float64), d["masses"]
    x_fs, v_fs = float(d["x_fs"]), float(d["v_fs"])
    if abs(x_fs - v_fs) > 1e-9:
        raise SystemExit(f"positions every {x_fs} fs but velocities every {v_fs}: pass equal --frames-x-fs and -v-fs")
    dev = "cuda"
    calc = load_calc(teacher_name, dev, "float32", enable_cueq=True)
    vol = abs(float(np.linalg.det(cell)))
    arms = sorted({k.split("__")[0] for k in d.files if k.endswith("__x")})
    skip = int(round(args.skip_fs / x_fs))
    report = {"frames": str(path), "teacher": teacher_name, "atoms": int(len(numbers)), "dt_fs": x_fs,
              "plateau_fs": [args.tc_lo, args.tc_hi], "blocks": args.blocks, "arms": {},
              "when": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}
    for arm in arms:
        xs, vs = d[f"{arm}__x"][skip:], d[f"{arm}__v"][skip:]
        n = min(len(xs), len(vs))
        t0 = time.perf_counter()
        p = stress_series(calc, numbers, cell, xs[:n], vs[:n], masses, dev)
        sec = time.perf_counter() - t0
        comps = shear_components(p)
        ke = 0.5 * (masses[None, :, None] * vs[:n].astype(np.float64) ** 2).sum(axis=(1, 2))
        temp = float(2.0 * ke.mean() / (3 * len(numbers) * KB))
        run_all, c = green_kubo(comps, x_fs, vol, temp, args.tc_hi)
        lo, hi = int(round(args.tc_lo / x_fs)), int(round(args.tc_hi / x_fs))
        eta = float(run_all[lo:hi + 1].mean())
        per = []
        size = n // args.blocks
        for b in range(args.blocks):
            r, _ = green_kubo(comps[b * size:(b + 1) * size], x_fs, vol, temp, args.tc_hi)
            per.append(float(r[lo:hi + 1].mean()))
        se = float(np.std(per, ddof=1) / np.sqrt(len(per))) if len(per) > 1 else None
        report["arms"][arm] = {"eta_mPas": eta, "eta_blocks": per, "eta_se": se, "temperature_K": temp,
                               "frames": int(n), "span_ps": n * x_fs / 1e3, "stress_s": sec,
                               "running_mPas": [float(v) for v in run_all[:: max(1, int(round(50.0 / x_fs)))]],
                               "acf_0": float(c[0])}
        print(f"{arm}: eta {eta:.3f} mPa s (block SE {se if se is None else round(se, 3)}; blocks "
              f"{', '.join(f'{v:.2f}' for v in per)}) | T {temp:.1f} K | {n} frames, {n * x_fs / 1e3:.1f} ps | "
              f"stress {sec:.0f} s", flush=True)
    out = OUT / f"viscosity_{stem}.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=1))
    tmp.replace(out)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
