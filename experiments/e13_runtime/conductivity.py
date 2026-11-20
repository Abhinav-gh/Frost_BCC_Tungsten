"""Ionic conductivity of every arm of a runtime run, from its saved positions (further_expansions_beyond.md 6.7).

Transport is the most sensitive dynamical observable we can measure without new code: a held force that shifts how ions
move shows here before it shows in an RDF. Two estimates per arm, as Gigli et al. (2024, Li3PS4) require, since the
Nernst-Einstein estimate can fail badly in superionic conductors:
- the collective (Einstein) conductivity from the mean square displacement of the total charge displacement
  M(t) = sum_i q_i (r_i(t) - r_i(0)): sigma = <|M(t)|^2> / (6 V k_B T t) at long t. For a neutral cell M is unchanged by
  a uniform drift of the whole crystal, so no drift correction is needed (trap 32 does not apply);
- the Nernst-Einstein estimate from each species' tracer diffusion (centre-of-mass drift removed first), and the Haven
  ratio between the two.
Charges are formal oxidation states. Errors come from splitting the run into blocks (the slope fitted per block).

Usage: python experiments/e13_runtime/conductivity.py data/results/e13/frames_<run>_<tag>.npz [--fit 2 10] [--blocks 4]
       -> prints a table, writes data/results/e13/conductivity_<run>_<tag>.json. CPU only (LANE=check).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.data import atomic_masses, chemical_symbols

CHARGE = {"Li": 1, "Na": 1, "K": 1, "Ag": 1, "Cu": 1, "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2, "Zn": 2, "Al": 3, "La": 3,
          "Ti": 4, "Zr": 4, "Si": 4, "Ge": 4, "Sn": 4, "P": 5, "O": -2, "S": -2, "F": -1, "Cl": -1, "Br": -1, "I": -1}
E2_OVER_KB = 1.602176634e-19 ** 2 / 1.380649e-23  # e^2 / k_B in C^2 K / J
A2_PER_PS = 1e-20 / 1e-12  # 1 A^2/ps in m^2/s


def msd_curve(disp: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Mean of |d(t0 + lag) - d(t0)|^2 over time origins; disp (frames, 3) or (frames, n, 3) summed over the last axis."""
    out = []
    for lag in lags:
        d = disp[lag:] - disp[:-lag]
        out.append(float((d ** 2).sum(-1).mean()))
    return np.asarray(out)


def slope(t: np.ndarray, y: np.ndarray, lo: float, hi: float) -> float:
    sel = (t >= lo) & (t <= hi)
    return float(np.polyfit(t[sel], y[sel], 1)[0]) if sel.sum() >= 3 else float("nan")


def analyse(x: np.ndarray, numbers: np.ndarray, volume: float, temperature: float, x_fs: float, fit: tuple, blocks: int):
    q = np.array([CHARGE[chemical_symbols[z]] for z in numbers], dtype=float)
    if abs(q.sum()) > 1e-6:
        raise SystemExit(f"formal charges do not sum to zero ({q.sum()}): add the right oxidation states to CHARGE")
    m = np.array([atomic_masses[z] for z in numbers])
    t_ps = np.arange(len(x)) * x_fs / 1000.0
    lags = np.unique(np.linspace(1, len(x) // 2, 60).astype(int))
    tl = lags * x_fs / 1000.0
    prefactor = E2_OVER_KB * 1e-20 / 1e-12 / (6.0 * volume * 1e-30 * temperature)  # (e^2 A^2/ps) -> S/m
    m_dip = (q[None, :, None] * (x - x[:1])).sum(1)  # collective charge displacement, e A
    sigma = prefactor * slope(tl, msd_curve(m_dip, lags), *fit)
    # per block (independent stretches) for the error bar
    per_block = []
    n = len(x) // blocks
    for b in range(blocks):
        xb = x[b * n:(b + 1) * n]
        lb = np.unique(np.linspace(1, len(xb) // 2, 40).astype(int))
        md = (q[None, :, None] * (xb - xb[:1])).sum(1)
        per_block.append(prefactor * slope(lb * x_fs / 1000.0, msd_curve(md, lb), fit[0] / 2, min(fit[1], len(xb) // 2 * x_fs / 1000)))
    # tracer diffusion per species, centre-of-mass drift removed
    com = (m[None, :, None] * x).sum(1) / m.sum()
    xc = x - (com - com[:1])[:, None, :]
    species = {}
    sigma_ne = 0.0
    for z in np.unique(numbers):
        sel = numbers == z
        d = xc[:, sel] - xc[:1, sel]
        msd = np.array([float(((d[lag:] - d[:-lag]) ** 2).sum(-1).mean()) for lag in lags])
        dcoef = slope(tl, msd, *fit) / 6.0  # A^2/ps
        species[chemical_symbols[z]] = {"count": int(sel.sum()), "D_A2_per_ps": dcoef, "charge": float(q[sel][0])}
        sigma_ne += E2_OVER_KB * sel.sum() * q[sel][0] ** 2 * dcoef * A2_PER_PS / (volume * 1e-30 * temperature)
    block_sd = float(np.std(per_block, ddof=1) / np.sqrt(blocks)) if blocks > 1 else float("nan")
    return {"sigma_S_per_m": sigma, "sigma_block_values": per_block, "sigma_block_se": block_sd,
            "sigma_nernst_einstein_S_per_m": float(sigma_ne), "haven_ratio": float(sigma_ne / sigma) if sigma > 0 else float("nan"),
            "species": species, "time_ps": float(t_ps[-1])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames")
    ap.add_argument("--fit", nargs=2, type=float, default=(2.0, 10.0), help="lag window (ps) for the slope")
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0, help="K; default from the run name (_T<K>)")
    a = ap.parse_args()
    path = Path(a.frames)
    d = np.load(path)
    stem = path.stem.removeprefix("frames_")
    temperature = a.temperature or float(stem.split("_T")[1].split("_")[0])
    cell = np.asarray(d["cell"], dtype=float)
    volume = float(abs(np.linalg.det(cell)))
    out = {"run": stem, "temperature_K": temperature, "volume_A3": volume, "fit_ps": list(a.fit), "arms": {}}
    for key in d.files:
        if key.endswith("__x"):
            out["arms"][key[:-3]] = analyse(np.asarray(d[key], dtype=np.float64), np.asarray(d["numbers"]), volume,
                                            temperature, float(d["x_fs"]), tuple(a.fit), a.blocks)
    ref = out["arms"].get("teacher")
    for arm, r in out["arms"].items():
        if ref:
            r["sigma_ratio_to_teacher"] = r["sigma_S_per_m"] / ref["sigma_S_per_m"] if ref["sigma_S_per_m"] else float("nan")
        print(f"{arm:12s} sigma {r['sigma_S_per_m']:9.3f} +- {r['sigma_block_se']:.3f} S/m | NE {r['sigma_nernst_einstein_S_per_m']:9.3f} "
              f"| Haven {r['haven_ratio']:.2f} | ratio {r.get('sigma_ratio_to_teacher', float('nan')):.3f} | D "
              + " ".join(f"{s} {v['D_A2_per_ps']:.4f}" for s, v in r["species"].items()))
    res = path.parent / f"conductivity_{stem}.json"
    res.write_text(json.dumps(out, indent=1))
    print("->", res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
