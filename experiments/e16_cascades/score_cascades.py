"""Pool cascade runs (cascade_runtime.py outputs) per arm and test them against the teacher, cascade by cascade.

    python experiments/e16_cascades/score_cascades.py data/results/e16/cascade_W16_c500.json data/results/e16/cascade_W16_c500loc_*.json

Per arm: surviving Frenkel pairs per cascade (mean, sd), the paired difference from the teacher arm of the same cascade
(the same seed: PKA atom, direction, border noise) with its standard error and sign counts, the peak damage and its time,
ms/fs, teacher calls and rollbacks. For held arms with a staleness trace: in how many cascades the worst region exceeded
epsilon in the ballistic phase (first --ballistic-fs) and when it first did, intervals that stood over a bound at k 1 and
intervals run on the teacher. Where damage traces exist: the mean damage and core temperature curves per arm on the
common sampling grid, and their mean absolute difference from the teacher's over the first 2 ps.
Output: data/results/e16/cascade_score_<--name>.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "data" / "results" / "e16"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--epsilon", type=float, default=2.0)
    ap.add_argument("--ballistic-fs", type=float, default=500.0)
    ap.add_argument("--name", default="pool")
    args = ap.parse_args()
    cascades = []
    for f in args.files:
        d = json.loads(Path(f).read_text())
        for c in d["cascades"]:
            cascades.append((Path(f).stem, c))
    arms = sorted({a for _, c in cascades for a in c["arms"]}, key=lambda a: (a != "teacher", a))
    out = {"files": args.files, "cascades": len(cascades), "arms": {}}
    for arm in arms:
        rows = [(src, c) for src, c in cascades if arm in c["arms"]]
        fp = np.array([c["arms"][arm]["vacancies"] for _, c in rows], dtype=float)
        rec = {"cascades": len(rows), "frenkel_mean": float(fp.mean()), "frenkel_sd": float(fp.std(ddof=1)) if len(fp) > 1 else 0.0,
               "ms_per_fs": float(np.mean([c["arms"][arm]["ms_per_fs"] for _, c in rows])),
               "teacher_calls": float(np.mean([c["arms"][arm]["teacher_calls"] for _, c in rows])),
               "rollbacks": float(np.mean([c["arms"][arm]["rollbacks"] for _, c in rows]))}
        peaks = [(c["arms"][arm]["peak_vacancies"], c["arms"][arm]["peak_fs"]) for _, c in rows if "peak_vacancies" in c["arms"][arm]]
        if peaks:
            rec["peak_mean"] = float(np.mean([p[0] for p in peaks]))
            rec["peak_fs_mean"] = float(np.mean([p[1] for p in peaks]))
        if arm != "teacher":
            pair = [(c["arms"]["teacher"]["vacancies"], c["arms"][arm]["vacancies"],
                     c["arms"]["teacher"]["ms_per_fs"] / c["arms"][arm]["ms_per_fs"]) for _, c in rows if "teacher" in c["arms"]]
            if pair:
                dd = np.array([b - a for a, b, _ in pair], dtype=float)
                rec["paired_diff"] = float(dd.mean())
                rec["paired_se"] = float(dd.std(ddof=1) / np.sqrt(len(dd))) if len(dd) > 1 else 0.0
                rec["lower_equal_higher"] = [int((dd < 0).sum()), int((dd == 0).sum()), int((dd > 0).sum())]
                rec["speedup_mean"] = float(np.mean([s for _, _, s in pair]))
            traces = [c["arms"][arm].get("stale_trace") for _, c in rows]
            traces = [t for t in traces if t]
            if traces:
                over, first, k1 = 0, [], 0
                for tr in traces:
                    held = [s for s in tr if s[2] is not None]
                    hits = [s[0] for s in held if s[0] < args.ballistic_fs and s[3] > args.epsilon]
                    if hits:
                        over += 1
                        first.append(hits[0])
                    k1 += sum(1 for s in tr if s[5] in ("o", "x"))
                rec["ballistic_local_over"] = f"{over}/{len(traces)}"
                rec["first_over_fs_mean"] = float(np.mean(first)) if first else None
                rec["k1_over_or_exact_per_cascade"] = k1 / len(traces)
                rec["exact_intervals"] = float(np.mean([c["arms"][arm].get("exact_intervals", 0) for _, c in rows]))
        dmg = [c["arms"][arm].get("damage_trace") for _, c in rows]
        dmg = [np.array(t, dtype=float) for t in dmg if t]
        if dmg:
            n = min(len(t) for t in dmg)
            stack = np.stack([t[:n] for t in dmg])
            rec["curve_fs"] = stack[0, :, 0].tolist()
            rec["curve_vacancies"] = stack[:, :, 1].mean(0).tolist()
            rec["curve_core_K"] = stack[:, :, 3].mean(0).tolist()
        out["arms"][arm] = rec
    ref = out["arms"].get("teacher", {})
    if "curve_fs" in ref:
        t_ref = np.array(ref["curve_fs"])
        for arm, rec in out["arms"].items():
            if arm == "teacher" or "curve_fs" not in rec:
                continue
            m = min(len(t_ref), len(rec["curve_fs"]))
            w = t_ref[:m] < 2000.0
            rec["curve_vac_mad_2ps"] = float(np.abs(np.array(rec["curve_vacancies"][:m]) - np.array(ref["curve_vacancies"][:m]))[w].mean())
            rec["curve_core_K_mad_2ps"] = float(np.abs(np.array(rec["curve_core_K"][:m]) - np.array(ref["curve_core_K"][:m]))[w].mean())
    OUT.mkdir(parents=True, exist_ok=True)
    res = OUT / f"cascade_score_{args.name}.json"
    res.write_text(json.dumps(out, indent=1))
    for arm, r in out["arms"].items():
        line = (f"{arm:9s} {r['cascades']:3d} cascades | Frenkel {r['frenkel_mean']:.2f} (sd {r['frenkel_sd']:.2f})"
                + (f" | paired {r['paired_diff']:+.2f} +- {r['paired_se']:.2f} (lower/equal/higher {r['lower_equal_higher']})"
                   if "paired_diff" in r else "")
                + (f" | peak {r['peak_mean']:.1f} at {r['peak_fs_mean']:.0f} fs" if "peak_mean" in r else "")
                + f" | {r['ms_per_fs']:.1f} ms/fs, calls {r['teacher_calls']:.0f}, rollbacks {r['rollbacks']:.1f}"
                + (f" | {r['speedup_mean']:.2f}x" if "speedup_mean" in r else "")
                + (f" | ballistic local over {r['ballistic_local_over']}, k1 over/exact {r['k1_over_or_exact_per_cascade']:.1f}"
                   if "ballistic_local_over" in r else "")
                + (f" | curves: vacancies {r['curve_vac_mad_2ps']:.2f}, core {r['curve_core_K_mad_2ps']:.1f} K"
                   if "curve_vac_mad_2ps" in r else ""))
        print(line)
    print("->", res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
