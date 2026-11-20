"""Point-defect observables for runs of a BCC metal with a vacancy or a self-interstitial (gaps.md E.8).

A single defect is invisible to the whole-cell spectrum and RDF that ``score_frames.py`` uses (one site in a thousand),
so a multirate run could create spurious defects or freeze the defect's motion and still pass. This scores what the
defect does, for every arm of a runtime run against its paired teacher arm:
- Wigner-Seitz occupancy of the perfect lattice in every kept frame (positions every 20 fs): the number of vacant and
  multiply occupied sites, and the fraction of frames holding exactly the defect the system started with (spurious
  Frenkel pairs or an annihilated defect show here);
- the defect's track (the vacant site, or the doubly occupied site), unwrapped across the box: hops (changes of site)
  and its diffusion coefficient from the mean square displacement.

The perfect lattice is ``frost.systems.bcc`` for the element, sized from the run's own cell (NVT keeps the volume).

Usage: python experiments/e13_runtime/defects.py data/results/e13/frames_<run>_<tag>.npz
       -> prints a table and writes data/results/e13/defects_<run>_<tag>.json
"""

from __future__ import annotations

import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from ase.data import chemical_symbols
from scipy.spatial import cKDTree

from frost.systems import BCC_A, bcc


def track(x: np.ndarray, sites: np.ndarray, box: np.ndarray, x_fs: float):
    """Occupancy and the defect site per frame; x (frames, atoms, 3), sites (n_sites, 3), orthorhombic box (3,)."""
    # The whole crystal moves: under Langevin friction its centre of mass random-walks (1,024 Fe atoms at 1/ps: ~2.3 A rms
    # over 100 ps), and the start frame sits wherever equilibration left it. Against a fixed ideal lattice that drift moves
    # every atom toward a cell boundary at once and the occupancy scrambles (tens of "vacancies" in every arm, the teacher
    # included; found 2026-11-13). So: align the lattice to the first frame (median minimum-image offset of its atoms from
    # their nearest sites), and remove each frame's centre-of-mass displacement from the first (exact on unwrapped frames).
    tree0 = cKDTree(np.mod(sites, box), boxsize=box)
    _, idx0 = tree0.query(np.mod(x[0], box))
    d0 = x[0] - sites[idx0]
    d0 -= box * np.round(d0 / box)
    sites = sites + np.median(d0, axis=0)
    tree = cKDTree(np.mod(sites, box), boxsize=box)
    com0 = x[0].mean(axis=0)
    n_vac, n_int, where = [], [], []
    for frame in x:
        frame = frame - (frame.mean(axis=0) - com0)
        _, idx = tree.query(np.mod(frame, box))
        occ = np.bincount(idx, minlength=len(sites))
        vac, mult = np.flatnonzero(occ == 0), np.flatnonzero(occ >= 2)
        n_vac.append(len(vac))
        n_int.append(int((occ[mult] - 1).sum()))
        where.append(vac[0] if len(vac) == 1 and len(mult) == 0 else (mult[0] if len(mult) == 1 and len(vac) == 0 else -1))
    where = np.asarray(where)
    ok = where >= 0
    hops = int((np.diff(where[ok]) != 0).sum()) if ok.sum() > 1 else 0
    # unwrapped defect path from consecutive minimum-image site displacements
    path = [np.zeros(3)]
    good = np.flatnonzero(ok)
    for a, b in itertools.pairwise(good):
        d = sites[where[b]] - sites[where[a]]
        d -= box * np.round(d / box)
        path.append(path[-1] + d)
    path = np.asarray(path)
    D = float("nan")
    if len(path) > 20:
        t = good * x_fs / 1000.0  # ps
        lags = np.arange(1, len(path) // 4)
        msd = np.array([((path[lag:] - path[:-lag]) ** 2).sum(1).mean() for lag in lags])
        dt = np.median(np.diff(t)) * lags
        D = float(np.polyfit(dt, msd, 1)[0] / 6.0)  # A^2/ps
    return {"frames": len(x), "frames_with_the_defect": float(ok.mean()), "mean_vacancies": float(np.mean(n_vac)),
            "mean_interstitials": float(np.mean(n_int)), "max_vacancies": int(max(n_vac)),
            "max_interstitials": max(n_int), "hops": hops, "D_defect_A2_per_ps": D}


def main(argv) -> int:
    path = Path(argv[0])
    d = np.load(path)
    numbers = d["numbers"]
    symbol = chemical_symbols[Counter(numbers.tolist()).most_common(1)[0][0]]
    box = np.diag(d["cell"]).astype(float)
    n = round(float(box[0]) / BCC_A[symbol])
    sites = bcc(symbol, n).get_positions() * (box / np.diag(bcc(symbol, n).cell.array))
    out = {"system": symbol, "sites": len(sites), "atoms": len(numbers), "arms": {}}
    for k in d.files:
        if k.endswith("__x"):
            out["arms"][k[:-3]] = track(np.asarray(d[k], dtype=np.float64), sites, box, float(d["x_fs"]))
    stem = path.stem.removeprefix("frames_")
    res = path.parent / f"defects_{stem}.json"
    res.write_text(json.dumps(out, indent=1))
    t = out["arms"].get("teacher", {})
    for arm, r in out["arms"].items():
        print(f"{arm:12s} defect present {r['frames_with_the_defect']:.3f} | vac {r['mean_vacancies']:.2f} "
              f"(max {r['max_vacancies']}) int {r['mean_interstitials']:.2f} (max {r['max_interstitials']}) | hops "
              f"{r['hops']} | D {r['D_defect_A2_per_ps']:.4f} A2/ps"
              + (f" (teacher {t.get('D_defect_A2_per_ps', float('nan')):.4f})" if arm != "teacher" else ""))
    print("->", res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
