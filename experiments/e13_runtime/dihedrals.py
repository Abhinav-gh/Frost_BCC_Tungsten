"""Backbone dihedrals of a solvated alanine dipeptide: populations, transitions and the (phi, psi) free-energy surface of
every arm in a frames file (further_expansions_beyond.md 6.2 and 6.3: the rate test in solution).

    PYTHONPATH=src python experiments/e13_runtime/dihedrals.py data/results/e13/frames_ala2off_T300_<tag>.npz [...]
    PYTHONPATH=src python experiments/e13_runtime/dihedrals.py --copies 64 frames_ala2gas_T300_<tag>.npz   (many copies)

Several frames files of one system (independent seeds) pool into one verdict per arm. ``--copies N``: the system holds N
dipeptides of 22 atoms each, contiguous (``frost.systems.ala2_gas``); every copy is analysed and they pool as independent
runs, and the teacher's own floor is the histogram L1 between its two halves of copies (independent molecules).

States are core sets (Buchete and Hummer 2008's transition-based assignment): a frame inside a core joins that state, a
frame between cores keeps the last core visited, and a transition is counted when the trajectory enters a core other than
its last one, so recrossings of a dividing line are not counted as transitions. Cores (degrees; phi from C-N-CA-C, psi
from N-CA-C-N, the dipeptide's first 22 atoms in ``frost.systems.ALA2``):
    beta / PII  phi < -45 and (psi > 100 or psi < -150)     extended, the most populated basin in water
    alphaR      phi < -45 and -80 < psi < 20                right-handed helix
    alphaL      phi > 30 and -20 < psi < 90                 left-handed helix, rare at 300 K
Outputs per arm: core populations, committed-state populations and the free-energy difference
dG(alphaR - beta) = -kT ln(p_alphaR / p_beta), transition counts and mean dwell times, and the L1 distance of the
(phi, psi) histogram (36 x 36 bins) from the teacher's. For two seeds of the teacher the same L1 is its own floor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from frost.systems import ALA2_PHI, ALA2_PSI

KB_KCAL = 0.0019872043  # kcal/mol/K
CORES = ("beta", "alphaR", "alphaL")


def dihedral(x: np.ndarray, idx, box: np.ndarray) -> np.ndarray:
    """Dihedral (degrees, -180..180) of atoms idx over frames x (F, N, 3), bond vectors by minimum image."""
    a, b, c, d = (x[:, i] for i in idx)

    def mic(v):
        return v - box * np.round(v / box)

    b0, b1, b2 = mic(a - b), mic(c - b), mic(d - c)
    b1n = b1 / np.linalg.norm(b1, axis=1, keepdims=True)
    v = b0 - (b0 * b1n).sum(1, keepdims=True) * b1n
    w = b2 - (b2 * b1n).sum(1, keepdims=True) * b1n
    xx = (v * w).sum(1)
    yy = (np.cross(b1n, v) * w).sum(1)
    return np.degrees(np.arctan2(yy, xx))


def core_of(phi: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Core index per frame (0 beta, 1 alphaR, 2 alphaL) or -1 between cores."""
    out = np.full(phi.shape, -1, dtype=np.int64)
    out[(phi < -45) & ((psi > 100) | (psi < -150))] = 0
    out[(phi < -45) & (psi > -80) & (psi < 20)] = 1
    out[(phi > 30) & (psi > -20) & (psi < 90)] = 2
    return out


def committed(core: np.ndarray) -> np.ndarray:
    """Transition-based assignment: each frame takes the last core visited (frames before the first core get -1)."""
    out = core.copy()
    last = -1
    for k in range(len(out)):
        if out[k] >= 0:
            last = out[k]
        else:
            out[k] = last
    return out


def analyse(x: np.ndarray, box: np.ndarray, x_fs: float, temperature: float) -> dict:
    return analyse_angles(dihedral(x, ALA2_PHI, box), dihedral(x, ALA2_PSI, box), x_fs, temperature)


def analyse_angles(phi: np.ndarray, psi: np.ndarray, x_fs: float, temperature: float) -> dict:
    core = core_of(phi, psi)
    state = committed(core)
    valid = state >= 0
    transitions = {}
    dwell = {c: [] for c in CORES}
    start = 0
    for k in range(1, len(state)):
        if state[k] != state[k - 1] and state[k - 1] >= 0:
            key = f"{CORES[state[k - 1]]}->{CORES[state[k]]}"
            transitions[key] = transitions.get(key, 0) + 1
            dwell[CORES[state[k - 1]]].append((k - start) * x_fs / 1e3)
            start = k
    pop_core = {c: float((core == i).mean()) for i, c in enumerate(CORES)}
    pop_state = {c: float((state[valid] == i).mean()) if valid.any() else 0.0 for i, c in enumerate(CORES)}
    kt = KB_KCAL * temperature
    pb, pa = pop_state["beta"], pop_state["alphaR"]
    dg = float(-kt * np.log(pa / pb)) if pa > 0 and pb > 0 else None
    hist, _, _ = np.histogram2d(phi, psi, bins=36, range=[[-180, 180], [-180, 180]])
    return {"frames": len(phi), "ps": len(phi) * x_fs / 1e3, "populations_core": pop_core,
            "populations_committed": pop_state, "dG_alphaR_minus_beta_kcal": dg, "transitions": transitions,
            "n_transitions": int(sum(transitions.values())),
            "mean_dwell_ps": {c: float(np.mean(v)) if v else None for c, v in dwell.items()},
            "hist": (hist / hist.sum()).tolist(), "phi_psi_every": [phi[::50].round(1).tolist(), psi[::50].round(1).tolist()]}


def pool(results: list[dict]) -> dict:
    """Pool independent runs of one arm: frames-weighted populations, summed transitions, frames-weighted histogram."""
    w = np.array([r["frames"] for r in results], dtype=np.float64)
    hist = sum(np.asarray(r["hist"]) * wi for r, wi in zip(results, w, strict=True)) / w.sum()
    pops = {c: float(sum(r["populations_committed"][c] * wi for r, wi in zip(results, w, strict=True)) / w.sum())
            for c in CORES}
    trans = {}
    for r in results:
        for k, v in r["transitions"].items():
            trans[k] = trans.get(k, 0) + v
    ps = float(sum(r["ps"] for r in results))
    n = int(sum(trans.values()))
    return {"runs": len(results), "ps": ps, "populations_committed": pops, "transitions": trans, "n_transitions": n,
            "transitions_per_ns": 1e3 * n / ps if ps else None, "hist": hist}


def main(argv) -> int:
    copies = 1
    if "--copies" in argv:
        i = argv.index("--copies")
        copies = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    paths = [Path(a) for a in argv]
    per_arm: dict[str, list] = {}
    out = {"files": [str(p) for p in paths], "runs": {}}
    for path in paths:
        d = np.load(path)
        box = np.diag(d["cell"]).astype(np.float64)
        temperature = float(path.stem.split("_T")[1].split("_")[0])
        x_fs = float(d["x_fs"])
        out["runs"][path.stem] = {}
        for k in d.files:
            if k.endswith("__x"):
                arm = k[:-3]
                x = np.asarray(d[k], dtype=np.float64)
                rs = [analyse_angles(dihedral(x, [i + 22 * c for i in ALA2_PHI], box),
                                     dihedral(x, [i + 22 * c for i in ALA2_PSI], box), x_fs, temperature)
                      for c in range(copies)]
                r = pool(rs) if copies > 1 else rs[0]
                out["runs"][path.stem][arm] = {kk: vv for kk, vv in r.items() if kk != "hist"}
                per_arm.setdefault(arm, []).extend(rs)
    pooled = {arm: pool(rs) for arm, rs in per_arm.items()}
    ref = pooled.get("teacher")
    out["pooled"] = {}
    for arm, p in pooled.items():
        row = {k: v for k, v in p.items() if k != "hist"}
        if ref is not None:
            row["hist_L1_vs_teacher"] = float(np.abs(p["hist"] - ref["hist"]).sum())
        out["pooled"][arm] = row
    # the teacher's own floor: L1 between its runs (two or more independent seeds)
    if copies == 1 and "teacher" in per_arm and len(per_arm["teacher"]) >= 2:
        hs = [np.asarray(r["hist"]) for r in per_arm["teacher"]]
        out["teacher_self_L1"] = [float(np.abs(hs[a] - hs[b]).sum()) for a in range(len(hs)) for b in range(a + 1, len(hs))]
    if copies > 1 and "teacher" in per_arm:  # the floor from independent molecules: two halves of the copies
        rs = per_arm["teacher"]
        h = len(rs) // 2
        out["teacher_copy_halves_L1"] = float(np.abs(pool(rs[:h])["hist"] - pool(rs[h:])["hist"]).sum())
        for arm, rr in per_arm.items():
            if arm != "teacher":
                hh = len(rr) // 2
                out["pooled"][arm]["halves_L1_vs_teacher"] = [
                    float(np.abs(pool(rr[:hh])["hist"] - pool(rs[:h])["hist"]).sum()),
                    float(np.abs(pool(rr[hh:])["hist"] - pool(rs[h:])["hist"]).sum())]
    stem = paths[0].stem.removeprefix("frames_") + ("_pooled" if len(paths) > 1 else "")
    res = paths[0].parent / f"dihedrals_{stem}.json"
    res.write_text(json.dumps(out, indent=1))
    for arm, row in out["pooled"].items():
        pc = row["populations_committed"]
        print(f"{arm:12s} {row['ps']:7.1f} ps | beta {pc['beta']:.3f} alphaR {pc['alphaR']:.3f} alphaL {pc['alphaL']:.3f} | "
              f"transitions {row['n_transitions']} ({row['transitions_per_ns'] or 0:.1f}/ns)"
              + (f" | hist L1 {row['hist_L1_vs_teacher']:.3f}" if "hist_L1_vs_teacher" in row else ""))
    if "teacher_self_L1" in out:
        print("teacher self L1:", [round(v, 3) for v in out["teacher_self_L1"]])
    if "teacher_copy_halves_L1" in out:
        print(f"teacher copy halves L1: {out['teacher_copy_halves_L1']:.3f}")
    print("->", res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
