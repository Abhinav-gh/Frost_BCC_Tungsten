"""Judge a long run against independent realisations, not against one control run.

A run's own window-to-window spread misses any mode slower than the window: silica's steady-state "failure" at k 10-12
(2026-11-03) was scored that way and was a false alarm (CLAIMS.md, 2026-11-05). Written for the pre-registered slow80 /
slowpet tests: is the late-window RDF of each multirate arm outside the reference arm's across-realisation spread?

For each realisation (frames file) and arm, the total RDF (frost.observables.rdf, the fidelity gate's estimator, every
100 fs) and the mean Si-O-Si angle, in windows of WIN ps. Per window:
- teacher floor: each teacher realisation against the mean of the other teacher realisations (L1, the gate's metric);
- each multirate arm against the mean of the teacher realisations other than its own (same footing as the floor);
- for comparison, the within-realisation half-split spread the earlier verdicts used (teacher's two halves of the
  window).
A multirate arm is outside the teacher's range if its distance exceeds the largest teacher-against-others distance.

usage: REF_ARM=teacher|fixed6 python experiments/e13_runtime/realisations.py WIN_PS FRAMES.npz [FRAMES.npz ...]
(writes realisations_report.json next to this script; copy it to data/results/e13/)
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

from frost.observables import rdf, vdos

KB = 8.617333262e-5  # eV/K; kept velocities are ASE units, so m v^2 (amu) is eV


def l1(a, b) -> float:
    return float(np.abs(b - a).sum() / np.abs(a).sum())


def angle_mean(x, numbers, cell):
    """Mean Si-O-Si angle (each O and its two nearest Si), degrees; NaN for a cell without both Si and O."""
    inv = np.linalg.inv(cell)
    si, o = np.where(numbers == 14)[0], np.where(numbers == 8)[0]
    if len(si) < 2 or len(o) == 0:
        return float("nan")
    d = x[si][None] - x[o][:, None]
    f = d @ inv
    f -= np.round(f)
    d = f @ cell
    r = np.linalg.norm(d, axis=-1)
    nn = np.argsort(r, axis=1)[:, :2]
    v1, v2 = d[np.arange(len(o)), nn[:, 0]], d[np.arange(len(o)), nn[:, 1]]
    c = (v1 * v2).sum(-1) / np.linalg.norm(v1, axis=-1) / np.linalg.norm(v2, axis=-1)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))).mean())


def group(name: str) -> str:
    """Realisations that share a Langevin seed are not independent: the shared noise pulls same-seed runs from different
    start frames onto one path (Cu 500: 0.000 A apart after 20 ps; silica stays 0.2-0.6 A apart, but its slow collective
    mode may still be driven in step). References and floors therefore exclude every run with the same seed (``_r<seed>``
    in the tag); GROUP_BY=none restores leave-one-out."""
    if os.environ.get("GROUP_BY", "seed") == "none":
        return name
    m = re.search(r"_r(\d+)$", name)
    return m.group(1) if m else name


def g_total(seg, every, cell, types, n_species, cells=None):
    """The fidelity gate's total RDF of a stretch of position frames. ``cells`` (frames, 3, 3), under NPT, gives each
    frame its own cell (2026-11-14): unwrapped positions measured with another frame's cell are wrong across the
    boundary (trap 28), as in ``score_frames.structure``."""
    if cells is None:
        return np.asarray(
            rdf(torch.as_tensor(seg[::every], dtype=torch.float64), cell, types, n_species)[
                "g_total"
            ]
        )
    gs = [np.asarray(rdf(torch.as_tensor(seg[i : i + 1], dtype=torch.float64), cells[i], types, n_species)["g_total"])
          for i in range(0, len(seg), every)]
    return np.mean(gs, axis=0)


def main():
    win_ps = float(sys.argv[1])
    REF = os.environ.get(
        "REF_ARM", "teacher"
    )  # the reference arm pooled over realisations
    files = sys.argv[2:]
    data = {}  # realisation -> arm -> list of (g_total per window, half A, half B, angle)
    for f in files:
        z = np.load(f)
        numbers, cell = z["numbers"], z["cell"]
        species, types = np.unique(numbers, return_inverse=True)
        x_fs = float(z["x_fs"])
        per = round(win_ps * 1e3 / x_fs)
        every = max(1, round(100.0 / x_fs))
        name = Path(f).stem.removeprefix("frames_")
        data[name] = {}
        for arm in sorted({k.split("__")[0] for k in z.files if k.endswith("__x")}):
            x = z[f"{arm}__x"]
            cells_all = z[f"{arm}__cells"] if f"{arm}__cells" in z.files else None
            rows = []
            for w0 in range(0, x.shape[0] - per + 1, per):
                seg = x[w0 : w0 + per]
                cs = cells_all[w0 : w0 + per] if cells_all is not None else None
                half = per // 2
                rows.append(
                    {
                        "g": g_total(seg, every, cell, types, len(species), cs),
                        "a": g_total(seg[:half], every, cell, types, len(species),
                                     cs[:half] if cs is not None else None),
                        "b": g_total(seg[half:], every, cell, types, len(species),
                                     cs[half:] if cs is not None else None),
                        "angle": float(
                            np.mean(
                                [
                                    angle_mean(s.astype(np.float64), numbers,
                                               cs[q * 10] if cs is not None else cell)
                                    for q, s in enumerate(seg[::10])
                                ]
                            )
                        ),
                    }
                )
            v_fs = float(z["v_fs"])
            if (
                f"{arm}__v" in z.files and v_fs <= 20.0
            ):  # the spectrum needs dense velocities
                v, masses = z[f"{arm}__v"], z["masses"]
                per_v = round(win_ps * 1e3 / v_fs)
                for wi, r in enumerate(rows):
                    vseg = v[wi * per_v : (wi + 1) * per_v].astype(np.float64)
                    if len(vseg) < per_v:
                        continue
                    ke = 0.5 * (masses[None, :, None] * vseg**2).sum((1, 2))
                    r["T"] = float((2.0 * ke / (3.0 * len(masses) * KB)).mean())
                    _, spec, _ = vdos(
                        torch.as_tensor(vseg), v_fs, max_lag=min(1024, per_v // 2)
                    )
                    spec = np.asarray(spec)
                    r["s"] = spec / spec.max()
            data[name][arm] = rows
        print(f"read {name}: arms {list(data[name])}", flush=True)
    names = list(data)
    n_win = min(len(data[n][REF]) for n in names)
    report = {"win_ps": win_ps, "windows": []}
    for w in range(n_win):
        teach = {n: data[n][REF][w]["g"] for n in names}
        floor = {
            n: l1(
                np.mean([teach[m] for m in names if group(m) != group(n)], 0), teach[n]
            )
            for n in names
        }
        half = {n: l1(data[n][REF][w]["a"], data[n][REF][w]["b"]) for n in names}
        row = {
            "t0_ps": w * win_ps,
            "teacher_floor": floor,
            "teacher_half_split": half,
            "arms": {},
            "angle": {
                n: {a: data[n][a][w]["angle"] for a in data[n] if len(data[n][a]) > w}
                for n in names
            },
        }
        for n in names:
            ref = np.mean([teach[m] for m in names if group(m) != group(n)], 0)
            for arm, rows in data[n].items():
                if arm == REF or w >= len(rows) or len(rows) < n_win:
                    continue
                row["arms"].setdefault(arm, {})[n] = l1(ref, rows[w]["g"])
        if all("s" in data[n][REF][w] for n in names):
            spec = {n: data[n][REF][w]["s"] for n in names}
            temp = {n: data[n][REF][w]["T"] for n in names}
            row["vdos_floor"] = {
                n: l1(
                    np.mean([spec[m] for m in names if group(m) != group(n)], 0),
                    spec[n],
                )
                for n in names
            }
            row["T_floor"] = {
                n: temp[n] - np.mean([temp[m] for m in names if group(m) != group(n)])
                for n in names
            }
            row["vdos"], row["dT"] = {}, {}
            for n in names:
                ref_s = np.mean([spec[m] for m in names if group(m) != group(n)], 0)
                ref_t = np.mean([temp[m] for m in names if group(m) != group(n)])
                for arm, rows in data[n].items():
                    if arm == REF or w >= len(rows) or "s" not in rows[w]:
                        continue
                    row["vdos"].setdefault(arm, {})[n] = l1(ref_s, rows[w]["s"])
                    row["dT"].setdefault(arm, {})[n] = rows[w]["T"] - ref_t
                    row.setdefault("dT_paired", {}).setdefault(arm, {})[n] = (
                        rows[w]["T"] - temp[n]
                    )
        report["windows"].append(row)
        fmax = max(floor.values())
        print(
            f"\nwindow {w * win_ps:.0f}-{(w + 1) * win_ps:.0f} ps: {REF} vs other {REF} "
            f"{min(floor.values()):.4f}-{fmax:.4f} (mean {np.mean(list(floor.values())):.4f}); "
            f"teacher half-split {np.mean(list(half.values())):.4f}"
        )
        for arm, by in sorted(row["arms"].items()):
            vals = list(by.values())
            out = sum(v > fmax for v in vals)
            print(
                f"  {arm:<9} {np.mean(vals):.4f} (range {min(vals):.4f}-{max(vals):.4f}); outside {REF} range in "
                f"{out}/{len(vals)}"
            )
        if "vdos" in row:
            vmax = max(row["vdos_floor"].values())
            tf = list(row["T_floor"].values())
            print(
                f"  VDOS: {REF} vs other {REF} {min(row['vdos_floor'].values()):.4f}-{vmax:.4f}; "
                f"T: {REF} vs others {min(tf):+.2f} to {max(tf):+.2f} K"
            )
            for arm in sorted(row["vdos"]):
                vals, dts = (
                    list(row["vdos"][arm].values()),
                    list(row["dT"][arm].values()),
                )
                print(
                    f"  {arm:<9} VDOS {np.mean(vals):.4f} (outside {sum(x > vmax for x in vals)}/{len(vals)}) | "
                    f"T {np.mean(dts):+.2f} K (range {min(dts):+.2f} to {max(dts):+.2f})"
                    f" | paired T {np.mean(list(row['dT_paired'][arm].values())):+.2f} K"
                )
        angs = {
            a: [row["angle"][n][a] for n in names if a in row["angle"][n]]
            for a in data[names[0]]
            if len(data[names[0]][a]) > w
        }
        print(
            "  mean Si-O-Si angle: "
            + "  ".join(
                f"{a} {np.mean(v):.2f}+-{np.std(v):.2f}" for a, v in angs.items()
            )
        )
    out = Path(__file__).with_name("realisations_report.json")
    out.write_text(json.dumps(report, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
