"""Fidelity of every arm of a runtime run against that run's own teacher arm (``md_runtime.py --frames``).

The same estimators as ``validate.py`` (``frost.observables``: windowed VDOS of the whole cell, RDF from unwrapped
frames with matching shifts, Einstein diffusion), applied identically to every arm, so the comparison never mixes
estimator versions (trap 28). The error bar is the teacher arm's own half-to-half spread; a pass needs the VDOS and
RDF distances at or under it. Temperature is from the kept velocities and the run's own (possibly repartitioned)
masses.

Usage
  .venvs/frost-mace/bin/python experiments/e13_runtime/score_frames.py data/results/e13/frames_<run>_<tag>.npz [SKIP_PS]
      [--species Z1,Z2,...]

--species scores only those elements' atoms (velocities, positions, masses), e.g. ``--species 1,8`` for the water film of
an interface (2026-11-15): in water on 72 layers of Cu, 2,557 of 2,898 atoms are Cu and the whole-cell VDOS and RDF say
little about the film, as a single defect is invisible to them (trap 32). The RDF is then the selected atoms' own
(pairs among them only); the output is score_<run>_<tag>_sp<Z1>-<Z2>.json.

SKIP_PS (optional) scores each Frost arm from SKIP_PS on, over the teacher arm's own length, against the whole teacher
arm: a long certified run's steady state after its probation climb (``md_runtime --fs`` longer than ``--teacher-fs``).
The comparison is then unpaired (the noise streams have parted), which the teacher's own half-to-half spread already
assumes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from frost.observables import msd, rdf, vdos

KB = 8.617333262e-5  # eV / K; the kept velocities are in ASE units, so 0.5 m v^2 is in eV


def spectrum(v: np.ndarray, v_fs: float) -> tuple[np.ndarray, np.ndarray]:
    freq, s, _ = vdos(torch.as_tensor(v, dtype=torch.float64), v_fs, max_lag=1024)
    keep = freq <= 150.0
    return freq[keep], s[keep] / s[keep].max()


def structure(x: np.ndarray, cell: np.ndarray, numbers: np.ndarray, every: int = 5, cells=None) -> np.ndarray:
    """The RDF over every ``every``-th frame; ``cells`` (frames, 3, 3), under NPT, gives each frame its own cell."""
    species, types = np.unique(numbers, return_inverse=True)
    if cells is None:
        return np.asarray(rdf(torch.as_tensor(x[::every], dtype=torch.float64), cell, types, len(species))["g_total"])
    gs = [np.asarray(rdf(torch.as_tensor(x[i:i + 1], dtype=torch.float64), cells[i], types, len(species))["g_total"])
          for i in range(0, len(x), every)]
    return np.mean(gs, axis=0)


def l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(b - a).sum() / np.abs(a).sum())


def observe(v, x, f, cells=None) -> dict:
    m = f["masses"][None, :, None]
    ke = 0.5 * (m * v.astype(np.float64) ** 2).sum((1, 2))
    freq, s = spectrum(v, float(f["v_fs"]))
    _, d = msd(torch.as_tensor(x, dtype=torch.float64), float(f["x_fs"]))
    return {"freq": freq, "vdos": s, "rdf": structure(x, f["cell"], f["numbers"], cells=cells),
            "T": float(np.mean(2.0 * ke / (3 * v.shape[1] * KB))), "D_A2_per_ps": float(d)}


def main(argv: list[str]) -> int:
    species = None
    if "--species" in argv:
        i = argv.index("--species")
        species = [int(z) for z in argv[i + 1].split(",")]
        argv = argv[:i] + argv[i + 2:]
    path = Path(argv[0])
    f = dict(np.load(path))
    if species is not None:
        keep = np.isin(f["numbers"], species)
        if not keep.any():
            raise SystemExit(f"no atoms of {species} in {path.name}")
        f["numbers"], f["masses"] = f["numbers"][keep], f["masses"][keep]
        for k in list(f):
            if k.endswith("__v") or k.endswith("__x"):
                f[k] = f[k][:, keep]
    arms = sorted({k.split("__")[0] for k in f if "__" in k})
    if "teacher" not in arms:
        raise SystemExit("no teacher arm in the frames file")
    tv, tx = f["teacher__v"], f["teacher__x"]
    tc = f.get("teacher__cells")  # NPT: each position frame's own cell
    ref = observe(tv, tx, f, tc)
    hv, hx = len(tv) // 2, len(tx) // 2
    a = observe(tv[:hv], tx[:hx], f, tc[:hx] if tc is not None else None)
    b = observe(tv[hv:], tx[hx:], f, tc[hx:] if tc is not None else None)
    own = {"vdos": l1(a["vdos"], np.interp(a["freq"], b["freq"], b["vdos"])), "rdf": l1(a["rdf"], b["rdf"])}
    print(f"{path.name}: teacher T {ref['T']:.1f} K, D {ref['D_A2_per_ps']:.3f} A2/ps, own spread VDOS "
          f"{own['vdos']:.4f} RDF {own['rdf']:.4f}")
    out = {"teacher": {"T": ref["T"], "D_A2_per_ps": ref["D_A2_per_ps"]}, "own_spread": own, "arms": {}}
    skip_ps = float(argv[1]) if len(argv) > 1 else 0.0
    if skip_ps:
        out["arm_skip_ps"] = skip_ps
    for arm in arms:
        if arm == "teacher":
            continue
        v, x, c = f[f"{arm}__v"], f[f"{arm}__x"], f.get(f"{arm}__cells")
        if skip_ps:
            sv, sx = round(1e3 * skip_ps / float(f["v_fs"])), round(1e3 * skip_ps / float(f["x_fs"]))
            v, x = v[sv:sv + len(tv)], x[sx:sx + len(tx)]
            c = c[sx:sx + len(tx)] if c is not None else None
            if len(v) < len(tv) or len(x) < len(tx):
                raise SystemExit(f"{arm}: shorter than skip + the teacher's length")
        o = observe(v, x, f, c)
        row = {"vdos_l1": l1(ref["vdos"], np.interp(ref["freq"], o["freq"], o["vdos"])), "rdf_l1": l1(ref["rdf"], o["rdf"]),
               "T_shift_K": o["T"] - ref["T"], "D_ratio": o["D_A2_per_ps"] / ref["D_A2_per_ps"] if ref["D_A2_per_ps"] else None}
        row["pass"] = row["vdos_l1"] <= own["vdos"] and row["rdf_l1"] <= own["rdf"]
        out["arms"][arm] = row
        print(f"  {arm}: VDOS {row['vdos_l1']:.4f} RDF {row['rdf_l1']:.4f} T {row['T_shift_K']:+.1f} K "
              f"D ratio {row['D_ratio']:.3f} -> {'pass' if row['pass'] else 'FAIL'}")
    suffix = f"_skip{skip_ps:g}" if skip_ps else ""
    if species is not None:
        suffix += "_sp" + "-".join(str(z) for z in species)
        out["species"] = species
    path.with_name(path.stem.replace("frames_", "score_") + suffix + ".json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
