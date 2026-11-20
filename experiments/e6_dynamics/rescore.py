"""Re-judge frost-only fidelity files against a reference written by the current code (trap 28).

A frost-only run (``validate.py --arms frost --teacher-ref ...``) stores its own observables, computed by the
estimators of the day it ran; its comparison columns were computed against whatever reference it was given.
When that reference predates an estimator fix (the unwrapped-frame RDF, the windowed VDOS), the comparison is
void but the run is not: this recomputes the same comparison as ``validate.py`` (L1 of the VDOS and of the total
RDF relative to the teacher's, plus the teacher's own half-to-half spread) against a current reference of the
same run, start and length.

Usage
  .venvs/frost-mace/bin/python experiments/e6_dynamics/rescore.py --ref data/results/e6/fidelity_cu500_T300_nve_ref50_v3.json \\
      data/results/e6/fidelity_cu500_T300_nve_f45_k32_50ps.json ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def l1(a, b, key, field=None):
    x = np.asarray(a[key][field] if field else a[key], dtype=float)
    y = np.asarray(b[key][field] if field else b[key], dtype=float)
    if key == "vdos" and x.shape != y.shape:
        y = np.interp(np.asarray(a["vdos_freq_THz"]), np.asarray(b["vdos_freq_THz"]), y)
    return float(np.abs(y - x).sum() / np.abs(x).sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True)
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    ref = json.loads(Path(args.ref).read_text())
    t, own = ref["teacher"], ref.get("teacher_self") or ref["teacher"].get("self")
    print(f"reference {Path(args.ref).name}: own spread VDOS {own['vdos_l1_relative']:.4f} RDF "
          f"{own['rdf_l1_relative']:.4f}, first moment {t['vdos_first_moment_THz']:.2f} THz")
    out = {}
    for run in args.runs:
        d = json.loads(Path(run).read_text())
        for key in ("run", "fs", "dt_fs"):
            if d[key] != ref[key]:
                raise SystemExit(f"{run}: {key} {d[key]} against the reference's {ref[key]}")
        if d["config"]["start"] != ref["config"]["start"]:
            raise SystemExit(f"{run}: start {d['config']['start']} against {ref['config']['start']}")
        f = d["frost"]
        row = {"vdos_l1": l1(t, f, "vdos"), "rdf_l1": l1(t, f, "rdf", "g_total"),
               "vdos_own": own["vdos_l1_relative"], "rdf_own": own["rdf_l1_relative"],
               "moment_shift_THz": f["vdos_first_moment_THz"] - t["vdos_first_moment_THz"],
               "temperature_shift_K": f["mean_temperature_K"] - t["mean_temperature_K"]}
        row["passes_structure_and_spectrum"] = row["vdos_l1"] <= row["vdos_own"] and row["rdf_l1"] <= row["rdf_own"]
        out[Path(run).name] = row
        print(f"{Path(run).name}: VDOS {row['vdos_l1']:.4f} (own {row['vdos_own']:.4f}) RDF {row['rdf_l1']:.4f} "
              f"(own {row['rdf_own']:.4f}) moment {row['moment_shift_THz']:+.3f} THz T {row['temperature_shift_K']:+.1f} K "
              f"-> {'pass' if row['passes_structure_and_spectrum'] else 'FAIL'}")
    dest = Path(args.ref).with_name(Path(args.ref).stem + "_rescored.json")
    dest.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
