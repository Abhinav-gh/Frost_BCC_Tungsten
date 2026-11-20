"""Calibration of the ThermostatBias clause: predicted against measured temperature shift, over every saved NVT run.

The clause predicts an arm's temperature bias from the shadow work it injects: dT = p / (3 gamma k_B), with p the
slope of the effective energy E - Q per atom (``eff_drift_meV_per_atom_per_ps``) and gamma the Langevin friction. Each
saved ``md_*.json`` with a teacher arm over the whole run gives, per multirate arm, the measured shift (arm mean T
minus the teacher arm's, same start and Langevin seed) and the predicted one (arm slope minus the teacher's).

How precise the measurement is depends on the system: same-seed runs stay synchronised on Cu, Li3PO4, MgO and Si, so
their paired difference is nearly noise-free, while silica and water drift apart and carry ~1 K of paired noise
(CLAUDE.md trap 30). The per-system noise floor is the scatter of the measured shift on control arms (predicted
|dT| < 0.15 K), and each system's verdict compares the residual (measured - predicted) with that floor.

Ramps (``melt3500*``) are left out: their target temperature moves, so no steady-state bias exists to predict.

Usage: python experiments/e13_runtime/bias_calibration.py  (CPU; writes data/results/e13/bias_calibration.json)
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
KB = 0.08617333  # meV/K
CONTROL = 0.15  # K: an arm predicted below this measures the system's paired noise
DUP = 0.005  # K: two arms this close on measured and predicted shift are one trajectory


def rows():
    """One row per distinct trajectory. Arms rerun under several tags with the same seed and start (the fixed controls,
    and adaptive arms whose clause versions made the same decisions) repeat one trajectory to GPU round-off (1e-4 K);
    two arms of one system agreeing within DUP on both numbers are kept once, or they would fake agreement and a zero
    noise floor."""
    out, seen = [], []
    for f in sorted(glob.glob(str(ROOT / "data" / "results" / "e13" / "md_*.json"))):
        d = json.loads(Path(f).read_text())
        c, a = d.get("config", {}), d.get("arms", {})
        name = Path(f).stem.removeprefix("md_")
        if c.get("ensemble") != "nvt" or "teacher" not in a or "melt" in name:
            continue
        te = a["teacher"]
        if (c.get("teacher_fs") or c.get("fs")) != c.get("fs") or "eff_drift_meV_per_atom_per_ps" not in te:
            continue  # the teacher arm must cover the whole run
        gamma = c["friction"] * 1000.0  # 1/ps
        for arm, v in a.items():
            if arm == "teacher" or not isinstance(v, dict) or "eff_drift_meV_per_atom_per_ps" not in v:
                continue
            if v.get("reached_fs", 0) < c["fs"]:
                continue
            pred = (v["eff_drift_meV_per_atom_per_ps"] - te["eff_drift_meV_per_atom_per_ps"]) / (3 * gamma * KB)
            sysname, meas = name.split("_T")[0], v["mean_T"] - te["mean_T"]
            if any(s == sysname and abs(m - meas) < DUP and abs(q - pred) < DUP for s, m, q in seen):
                continue
            seen.append((sysname, meas, pred))
            out.append(
                {
                    "run": name,
                    "system": name.split("_T")[0],
                    "arm": arm,
                    "ps": c["fs"] / 1000,
                    "mean_k": v.get("mean_k"),
                    "measured_K": v["mean_T"] - te["mean_T"],
                    "predicted_K": pred,
                }
            )
    return out


def main() -> int:
    data = rows()
    report = {"rows": data, "systems": {}}
    print(f"{len(data)} arms; residual = measured - predicted (K); noise = sd of measured on control arms")
    for s in sorted({r["system"] for r in data}):
        rs = [r for r in data if r["system"] == s]
        m = np.array([r["measured_K"] for r in rs])
        p = np.array([r["predicted_K"] for r in rs])
        ctl = m[np.abs(p) < CONTROL]
        noise = float(ctl.std(ddof=1)) if len(ctl) > 2 else float("nan")
        hot = p >= 0.5  # arms where the clause predicts a resolvable shift
        res = m - p
        ratio = float(m[hot].sum() / p[hot].sum()) if hot.any() else float("nan")
        worst = float(res[hot].max()) if hot.any() else float("nan")
        report["systems"][s] = {
            "arms": len(rs),
            "control_arms": len(ctl),
            "noise_K": noise,
            "residual_mean_K": float(res.mean()),
            "residual_sd_K": float(res.std(ddof=1)),
            "hot_arms": int(hot.sum()),
            "measured_over_predicted_hot": ratio,
            "worst_underprediction_hot_K": worst,
        }
        print(
            f"{s:14s} arms {len(rs):3d} | noise {noise:5.2f} K ({len(ctl)} controls) | residual {res.mean():+.2f}"
            f" +- {res.std(ddof=1):.2f} K | arms predicted >= 0.5 K: {int(hot.sum())}, measured/predicted {ratio:.2f},"
            f" worst under-prediction {worst:+.2f} K"
        )
    out = ROOT / "data" / "results" / "e13" / "bias_calibration.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=1))
    tmp.replace(out)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
