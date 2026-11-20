"""Each Frost arm's pinned <Q> against its own seed's teacher, judged by the seed-to-seed scatter (trap 30), not by block
errors (gaps.md, melting: the teacher's two 1,250 K seeds differed by 0.010 in <Q> against block errors of 0.002-0.003, so
an inverse-variance pool of block errors overstated every T_m difference's significance).

    PYTHONPATH=src python experiments/e17_melting/paired_q.py cuslab 1275 kdiag40 [--slope-tags final40,kdiag40]

Per seed: dQ = <Q>_arm - <Q>_teacher (both after the run's equilibration skip; the arms share the seed's start frame and
Langevin stream, so the pairing removes the common interface wander). Across seeds: the mean, its standard error from the
seeds' scatter, and a Student t interval. The T_m shift follows from the teacher's slope d<Q>/dT across temperatures
(every teacher run under --slope-tags): an arm whose <Q> is lower at the same T behaves as a hotter run, so
dT_m = -dQ / (d<Q>/dT) with d<Q>/dT < 0, i.e. dT_m has the sign of dQ.
Output: paired_q_<label>_T<T>_<tag>.json in data/results/e17.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
R13, R17 = ROOT / "data" / "results" / "e13", ROOT / "data" / "results" / "e17"
T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23}


def teacher_slope(label: str, tags: list[str]) -> tuple[float, float]:
    """d<Q>/dT from every teacher run of these tags: a weighted line through per-temperature seed means."""
    by_t: dict[float, list[float]] = {}
    for tag in tags:
        for path in R13.glob(f"md_{label}_T*_{tag}_s*.json"):
            m = re.search(r"_T(\d+)_", path.name)
            q = json.loads(path.read_text())["arms"]["teacher"]["pin"]["q_mean"]
            by_t.setdefault(float(m.group(1)), []).append(q)
    ts = np.array(sorted(by_t))
    qs = np.array([np.mean(by_t[t]) for t in ts])
    a = np.stack([np.ones_like(ts), ts], 1)
    coef, res, *_ = np.linalg.lstsq(a, qs, rcond=None)
    dof = max(len(ts) - 2, 1)
    s2 = float(res[0]) / dof if len(res) else 0.0
    cov = s2 * np.linalg.inv(a.T @ a)
    return float(coef[1]), float(math.sqrt(cov[1, 1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("label")
    ap.add_argument("temperature", type=int)
    ap.add_argument("tag")
    ap.add_argument("--slope-tags", default="final40,kdiag40")
    args = ap.parse_args()
    runs = sorted(R13.glob(f"md_{args.label}_T{args.temperature}_{args.tag}_s*.json"))
    if not runs:
        raise SystemExit("no runs")
    slope, slope_se = teacher_slope(args.label, args.slope_tags.split(","))
    per_arm: dict[str, list[float]] = {}
    teacher_q = []
    seeds = []
    for path in runs:
        arms = json.loads(path.read_text())["arms"]
        q0 = arms["teacher"]["pin"]["q_mean"]
        teacher_q.append(q0)
        seeds.append(path.stem.rsplit("_s", 1)[1])
        for arm, v in arms.items():
            if arm != "teacher" and "pin" in v:
                per_arm.setdefault(arm, []).append(v["pin"]["q_mean"] - q0)
    out = {"label": args.label, "T_K": args.temperature, "tag": args.tag, "seeds": seeds,
           "teacher_q": teacher_q, "teacher_q_seed_sd": float(np.std(teacher_q, ddof=1)) if len(teacher_q) > 1 else None,
           "slope_dQ_dT": slope, "slope_se": slope_se, "arms": {}}
    print(f"{args.label} {args.temperature} K, {len(runs)} seeds ({', '.join(seeds)}); teacher <Q> "
          f"{np.mean(teacher_q):.4f}, seed sd {out['teacher_q_seed_sd'] or float('nan'):.4f}; d<Q>/dT {slope:.2e} +- "
          f"{slope_se:.1e} per K")
    for arm, d in sorted(per_arm.items()):
        d = np.array(d)
        n = len(d)
        mean = float(d.mean())
        se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
        half = T95.get(n - 1, 2.0) * se if n > 1 else float("nan")
        # an arm with lower <Q> at T behaves as the teacher at T + dQ/slope (slope < 0); its melting point is lower by
        # that amount: dT_m = -(dQ / slope)
        dtm = -(mean / slope)
        dtm_half = abs(half / slope)
        out["arms"][arm] = {"dQ": d.tolist(), "dQ_mean": mean, "dQ_se": se, "dQ_95": half, "dTm_K": dtm,
                            "dTm_95_K": dtm_half, "excludes_zero": bool(abs(mean) > half)}
        print(f"  {arm:9s} dQ {mean:+.4f} +- {se:.4f} (95% +- {half:.4f}; per seed {', '.join(f'{x:+.4f}' for x in d)}) "
              f"-> dT_m {dtm:+.1f} +- {dtm_half:.1f} K (95%){'  EXCLUDES 0' if abs(mean) > half else ''}")
    R17.mkdir(parents=True, exist_ok=True)
    res = R17 / f"paired_q_{args.label}_T{args.temperature}_{args.tag}.json"
    tmp = res.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(res)
    print("->", res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
