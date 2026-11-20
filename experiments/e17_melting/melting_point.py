"""The melting point of every arm from pinned two-phase runs (``md_runtime.py --pin``, slabs from ``prep_slab.py``):
further_expansions_beyond.md 7.2.

    PYTHONPATH=src python experiments/e17_melting/melting_point.py cuslab pin50

reads data/results/e13/md_<label>_T<T>_<tag>*.json (every temperature, every seed), and for each arm:
- gap(T) = a - <Q> per run (<Q> after the run's equilibration, its error by block averaging), seeds pooled by inverse
  variance;
- T_m where gap crosses zero (a weighted straight line, ``frost.pinning.melting_point``), with its standard error;
- dmu(T) = kappa (Q_s - Q_l) gap / N (Q_s, Q_l from the slab's preparation) and the entropy of melting dS = d dmu / dT,
  so the latent heat L = T_m dS is a second, independent check of the arm against the teacher.
The verdict compares each Frost arm's T_m with the teacher's: the difference against their combined standard error.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np

from frost.pinning import melting_point

ROOT = Path(__file__).resolve().parents[2]
R13, R17 = ROOT / "data" / "results" / "e13", ROOT / "data" / "results" / "e17"
KB = 8.617333262e-5


def main(argv) -> int:
    label, tag = argv[0], argv[1]
    runs = sorted(R13.glob(f"md_{label}_T*_{tag}*.json"))
    if not runs:
        raise SystemExit(f"no runs md_{label}_T*_{tag}*.json")
    per_arm: dict[str, dict[float, list]] = {}
    speed: dict[str, list] = {}
    prep = {}
    for path in runs:
        temp = float(re.search(r"_T(\d+)_", path.name).group(1))
        d = json.loads(path.read_text())
        pj = R17 / f"{label}_T{int(temp)}_prep.json"
        if pj.exists():
            prep[temp] = json.loads(pj.read_text())
        for arm, r in d["arms"].items():
            p = r.get("pin", {})
            if "q_mean" not in p:
                continue
            per_arm.setdefault(arm, {}).setdefault(temp, []).append((p["gap"], p["q_se_block"], path.name))
            if "speedup" in r:
                speed.setdefault(arm, []).append(r["speedup"])
    out = {"label": label, "tag": tag, "runs": [p.name for p in runs], "arms": {}}
    for arm, by_t in per_arm.items():
        temps, gaps, ses = [], [], []
        for temp in sorted(by_t):
            g = np.array([v[0] for v in by_t[temp]])
            e = np.array([v[1] for v in by_t[temp]])
            w = 1.0 / np.maximum(e, 1e-9) ** 2
            temps.append(temp)
            gaps.append(float((w * g).sum() / w.sum()))
            # the larger of the propagated error and the seeds' own scatter (a block estimate can be optimistic)
            se_prop = float(1.0 / math.sqrt(w.sum()))
            se_seed = float(g.std(ddof=1) / math.sqrt(len(g))) if len(g) > 1 else 0.0
            ses.append(max(se_prop, se_seed))
        row = {"temperatures": temps, "gap": gaps, "gap_se": ses, "seeds": [len(by_t[t]) for t in sorted(by_t)]}
        if len(temps) >= 2:
            tm, tm_se = melting_point(temps, gaps, ses)
            row.update({"T_m": tm, "T_m_se": tm_se})
            p0 = prep.get(temps[0])
            if p0:
                scale = p0["kappa_eV"] * (p0["q_solid"] - p0["q_liquid"]) / p0["n_atoms"]
                slope = np.polyfit(temps, scale * np.asarray(gaps), 1, w=1.0 / np.asarray(ses))[0]  # eV/atom/K
                row.update({"dS_kB": float(slope / KB), "latent_heat_eV": float(slope * tm)})
        if arm in speed:
            row["speedup_mean"] = float(np.mean(speed[arm]))
        out["arms"][arm] = row
    ref = out["arms"].get("teacher", {})
    for arm, row in out["arms"].items():
        if arm != "teacher" and "T_m" in row and "T_m" in ref:
            diff = row["T_m"] - ref["T_m"]
            se = math.hypot(row["T_m_se"], ref["T_m_se"])
            row["dT_m_vs_teacher"], row["dT_m_sigma"] = diff, diff / se if se else None
    res = R17 / f"melting_{label}_{tag}.json"
    R17.mkdir(parents=True, exist_ok=True)
    res.write_text(json.dumps(out, indent=1))
    for arm, row in out["arms"].items():
        line = f"{arm:12s} " + " ".join(f"{t:.0f}K gap {g:+.4f}+-{e:.4f}" for t, g, e in
                                         zip(row["temperatures"], row["gap"], row["gap_se"], strict=True))
        if "T_m" in row:
            line += f" | T_m {row['T_m']:.1f} +- {row['T_m_se']:.1f} K"
        if "latent_heat_eV" in row:
            line += f" | L {row['latent_heat_eV']:.3f} eV/atom"
        if "dT_m_vs_teacher" in row:
            line += f" | vs teacher {row['dT_m_vs_teacher']:+.1f} K ({row['dT_m_sigma']:+.1f} sigma)"
        if "speedup_mean" in row:
            line += f" | {row['speedup_mean']:.2f}x"
        print(line)
    print("->", res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
