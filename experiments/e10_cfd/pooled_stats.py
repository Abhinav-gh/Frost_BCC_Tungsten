"""3D statistics pooled over independent DNS references, with uncertainties from the trajectories themselves.

One DNS reference of six trajectories is not a converged statistic for a saturated dynamo: two independent references
(``mhd256f.h5``, ``mhd256f_s3.h5``) differ by 12% in magnetic energy and 0.13 in magnetic spectrum, while each one's
half-against-half floor read 1-9%. A method scored against one reference inherits that reference's chance fluctuation, and
a closure trained against it learns it. Following the practice for DNS statistics (Oliver, Malaya, Ulerich and Moser,
Phys. Fluids 26, 035101, 2014: uncertainty from the realisations, not from one average) and for ML comparisons (Bouthillier
et al., MLSys 2021), this pools every reference's trajectories and every method's runs from every reference's starts, and
reports for each statistic:

- the pooled score (method mean against DNS mean, ``stats3d.scores_any``);
- a 95% bootstrap interval over trajectories (DNS and method trajectories resampled independently), so a signed error
  whose interval contains 0 is not a measured bias;
- the DNS floor at this sample size: the 95th percentile of the score between random halves of the pooled DNS
  trajectories (a conservative stand-in for two independent references of the pooled size).

A method *matches* a statistic when its bootstrap interval reaches the floor (spectra) or contains 0 (signed errors).

Input: stats3d JSON files written with per-trajectory statistics (``dns_traj_stats``, rows' ``traj_stats``), one per
reference, rows named alike. Usage
  PYTHONPATH=src .venvs/frost-mhd/bin/python experiments/e10_cfd/pooled_stats.py data/results/e10/stats3d_w5fix_{A,B,C}.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats3d import scores_any  # noqa: E402

KEYS = ("spectrum_l1", "spectrum_l1_high", "energy_err", "enstrophy_err", "flatness_band_err",
        "mag_spectrum_l1", "mag_spectrum_l1_high", "mag_energy_err", "mag_enstrophy_err", "mag_flatness_band_err")
UNSIGNED = {"spectrum_l1", "spectrum_l1_high", "mag_spectrum_l1", "mag_spectrum_l1_high"}


def main(argv: list[str]) -> int:
    files = [json.loads(Path(p).read_text()) for p in argv]
    kh, shells = files[0]["kh"], files[0].get("shells", 12)
    dns = np.concatenate([np.asarray(f["dns_traj_stats"]) for f in files])
    w = np.concatenate([np.full(len(f["dns_traj_stats"]), f["dns_records"], float) for f in files])
    rng = np.random.default_rng(0)

    def wmean(x, wt):
        return (x * wt[:, None]).sum(0) / wt.sum()

    ref = wmean(dns, w)
    # the DNS floor: random halves of the pooled DNS trajectories
    floor = {k: [] for k in KEYS}
    n = len(dns)
    for _ in range(400):
        perm = rng.permutation(n)
        a, b = perm[: n // 2], perm[n // 2:]
        sc = scores_any(wmean(dns[a], w[a]), wmean(dns[b], w[b]), kh, shells)
        for k in KEYS:
            floor[k].append(abs(sc[k]))
    floor95 = {k: float(np.percentile(v, 95)) for k, v in floor.items()}
    print(f"pooled DNS: {n} trajectories from {len(files)} references; floor (95th percentile, random halves): "
          + " ".join(f"{k}={v:.3f}" for k, v in floor95.items()))
    out = {"n_dns_trajectories": n, "floor95": floor95, "rows": {}}
    names = [r for r in files[0]["rows"] if all(r in f["rows"] and "traj_stats" in f["rows"][r] for f in files)]
    for name in names:
        runs = np.concatenate([np.asarray(f["rows"][name]["traj_stats"]) for f in files])
        point = scores_any(runs.mean(0), ref, kh, shells)
        boot = {k: [] for k in KEYS}
        for _ in range(400):
            i = rng.integers(0, len(runs), len(runs))
            j = rng.integers(0, n, n)
            sc = scores_any(runs[i].mean(0), wmean(dns[j], w[j]), kh, shells)
            for k in KEYS:
                boot[k].append(sc[k])
        row = {"ms_per_0.04": float(np.mean([f["rows"][name]["ms_per_0.04"] for f in files])), "n_runs": len(runs)}
        verdicts = []
        for k in KEYS:
            lo, hi = (float(x) for x in np.percentile(boot[k], [2.5, 97.5]))
            ok = (lo <= floor95[k]) if k in UNSIGNED else (lo <= 0.0 <= hi)
            row[k] = {"value": float(point[k]), "ci95": [lo, hi], "matches": bool(ok)}
            verdicts.append(ok)
        row["matches_all"] = bool(all(verdicts))
        out["rows"][name] = row
        print(f"{name:34s} {row['ms_per_0.04']:7.1f} ms | " + " ".join(
            f"{k.replace('_err', '').replace('spectrum_l1', 'spec')}={row[k]['value']:+.3f}{'' if row[k]['matches'] else '*'}"
            for k in KEYS) + f" | {'MATCHES ALL' if row['matches_all'] else ''}", flush=True)
    dest = Path(argv[0]).with_name("pooled_" + "_".join(Path(p).stem.replace("stats3d_", "") for p in argv) + ".json")
    dest.write_text(json.dumps(out, indent=1))
    print(f"-> {dest}  (* = outside the floor / interval excludes 0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
