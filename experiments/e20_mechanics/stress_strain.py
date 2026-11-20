"""Stress-strain curves of every arm of a uniaxial-tension run (``md_runtime.py --ensemble npt --npt-axes yz
--strain-rate R``): further_expansions_beyond.md 7.5 (mechanics, tier 3).

    PYTHONPATH=src python experiments/e20_mechanics/stress_strain.py data/results/e13/md_<run>_<tag>.json [...]

Per arm: true strain ln(L_x / L_x0) and tensile stress sigma_xx = -P_xx (GPa) from the teacher's exact virial at each box
move (lateral stresses beside it, ~0 under their barostats), binned in strain; the yield point (the peak tensile stress,
and the strain at it: homogeneous dislocation nucleation in a perfect crystal), the flow stress after it, and the curve's
mean absolute difference from the teacher's over the elastic range (strain under 80% of the teacher's yield strain).
Several files (seeds) pool per arm: yield values as mean and spread. Output: stress_strain_<first file's stem>.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

EV_A3_TO_GPA = 160.21766


def curve(trace, bin_width: float = 0.0025):
    t = np.asarray(trace, dtype=np.float64)
    lx, pxx, pyy, pzz = t[:, 1], t[:, 4], t[:, 5], t[:, 6]
    strain = np.log(lx / lx[0])
    sxx, lat = -pxx * EV_A3_TO_GPA, -0.5 * (pyy + pzz) * EV_A3_TO_GPA
    edges = np.arange(0.0, strain.max() + bin_width, bin_width)
    idx = np.digitize(strain, edges) - 1
    mids, s_mean, l_mean = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() >= 2:
            mids.append(0.5 * (edges[b] + edges[b + 1]))
            s_mean.append(float(sxx[m].mean()))
            l_mean.append(float(lat[m].mean()))
    mids, s_mean = np.array(mids), np.array(s_mean)
    peak = int(np.argmax(s_mean))
    after = s_mean[peak + 1:]
    return {"strain": mids.tolist(), "sigma_xx_GPa": s_mean.tolist(), "lateral_GPa": l_mean,
            "yield_stress_GPa": float(s_mean[peak]), "yield_strain": float(mids[peak]),
            "flow_stress_GPa": float(after[len(after) // 2:].mean()) if len(after) >= 4 else None,
            "max_strain": float(strain.max())}


def main(argv) -> int:
    per_arm: dict[str, list] = {}
    for f in argv:
        d = json.loads(Path(f).read_text())
        for arm, v in d["arms"].items():
            if v.get("stress_trace"):
                per_arm.setdefault(arm, []).append(curve(v["stress_trace"]) | {"speedup": v.get("speedup")})
    if "teacher" not in per_arm:
        raise SystemExit("no teacher stress trace")
    out = {"files": argv, "arms": {}}
    ref = per_arm["teacher"][0]
    elastic_max = 0.8 * ref["yield_strain"]
    for arm, cs in per_arm.items():
        ys = np.array([c["yield_stress_GPa"] for c in cs])
        ye = np.array([c["yield_strain"] for c in cs])
        row = {"runs": len(cs), "yield_stress_GPa": float(ys.mean()), "yield_stress_spread": float(ys.std()),
               "yield_strain": float(ye.mean()), "yield_strain_spread": float(ye.std()),
               "flow_stress_GPa": [c["flow_stress_GPa"] for c in cs], "speedup": [c["speedup"] for c in cs],
               "curves": cs}
        if arm != "teacher":
            diffs = []
            for c in cs:
                s = np.interp(ref["strain"], c["strain"], c["sigma_xx_GPa"])
                m = np.asarray(ref["strain"]) < elastic_max
                diffs.append(float(np.abs(s[m] - np.asarray(ref["sigma_xx_GPa"])[m]).mean()))
            row["elastic_mean_abs_diff_GPa"] = float(np.mean(diffs))
        out["arms"][arm] = row
    res = Path(argv[0]).parent / f"stress_strain_{Path(argv[0]).stem.removeprefix('md_')}.json"
    res.write_text(json.dumps(out, indent=1))
    for arm, row in out["arms"].items():
        print(f"{arm:10s} yield {row['yield_stress_GPa']:.3f} GPa at strain {row['yield_strain']:.4f}"
              + (f" (spread {row['yield_stress_spread']:.3f} GPa, {row['yield_strain_spread']:.4f})" if row["runs"] > 1 else "")
              + (f" | elastic |d sigma| {row['elastic_mean_abs_diff_GPa']:.4f} GPa" if "elastic_mean_abs_diff_GPa" in row else "")
              + (f" | {np.mean([x for x in row['speedup'] if x]):.2f}x" if any(row["speedup"]) else ""))
    print("->", res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
