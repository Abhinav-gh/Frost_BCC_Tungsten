"""Gap 1: the paired skill test pooled over several seasons of starts.

Each season's ``forecast_skill.py`` result holds, per variant, the RMSE of every start at every
6-hourly lead. The pooled test takes the relative paired difference start by start against the
released model, as ``forecast_skill.py`` does, and combines the seasons: within a season the
starts overlap (every 2 days, 10-day forecasts), so each season's effective sample size is deflated
by its own lag-1 correlation (``paired_interval``); seasons are months apart and independent, so
their means combine by inverse variance. A variant is certified when the upper end of the 95%
interval stays below the margin at every lead, for z500 and t850 both.

Output: data/results/e11/skill_pooled_<tag>.json

Usage
  .venvs/frost-gcm/bin/python experiments/e11_gcm/pool_skill.py --runs 2p8_jf,2p8_jj2020,2p8_on2020 --tag 2p8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast_skill import paired_interval  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "data" / "results" / "e11"
FIELDS = ("geopotential_500", "temperature_850")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", required=True, help="skill_<run>.json files, comma-separated")
    ap.add_argument("--variants", default="sub3+damp+f32,sub2+damp+f32")
    ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    seasons = [json.loads((RES / f"skill_{r}.json").read_text()) for r in args.runs.split(",")]
    out = {"runs": args.runs.split(","), "margin": args.margin, "variants": {}}
    for v in args.variants.split(","):
        verdict, rows = True, {}
        for field in FIELDS:
            means, variances, n_effs, n = [], [], [], 0
            for s in seasons:
                ref = np.asarray(s["rows"]["released"]["rmse"][field])
                d = (np.asarray(s["rows"][v]["rmse"][field]) - ref) / ref  # relative, per start and lead
                m, half, n_eff = paired_interval(d)
                t = scipy.stats.t.ppf(0.975, np.maximum(n_eff - 1.0, 1.0))
                means.append(m)
                variances.append((half / t) ** 2)
                n_effs.append(n_eff)
                n += d.shape[0]
            w = 1.0 / np.maximum(np.stack(variances), 1e-30)
            mean = (w * np.stack(means)).sum(0) / w.sum(0)
            se = 1.0 / np.sqrt(w.sum(0))
            dof = np.maximum(np.stack(n_effs).sum(0) - len(seasons), 1.0)
            upper = mean + scipy.stats.t.ppf(0.975, dof) * se
            fails = np.nonzero(upper > args.margin)[0]
            ok = len(fails) == 0
            verdict &= ok
            leads = [round(4 * d) - 1 for d in (1, 3, 5, 10)]
            rows[field] = {"n_starts": n, "mean_pct": (100 * mean).tolist(), "upper_pct": (100 * upper).tolist(),
                           "certified": ok, "first_failing_lead_h": (int(6 * (fails[0] + 1)) if len(fails) else None)}
            print(f"{v:18s} {field:18s} pooled over {n} starts: day 1/3/5/10 mean "
                  + "/".join(f"{100 * mean[i]:+.1f}" for i in leads) + "%  upper "
                  + "/".join(f"{100 * upper[i]:+.2f}" for i in leads) + "%  "
                  + ("certified" if ok else f"fails from {6 * (fails[0] + 1)} h"), flush=True)
        out["variants"][v] = {"certified": verdict, "fields": rows}
    (RES / f"skill_pooled_{args.tag}.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
