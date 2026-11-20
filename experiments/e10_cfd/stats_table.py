"""Print a ``stats3d.py`` result as a like-for-like table: every row re-scored from its stored mean
statistics against the DNS and, optionally, against a converged classical row (the second yardstick,
V.6), on the shells and gradient band chosen here. Flow and MHD files alike (MHD adds the magnetic
spectrum, current, the field's derivative flatness and cross helicity).

Usage
  env -u LD_LIBRARY_PATH -u CUDA_HOME JAX_PLATFORMS=cpu PYTHONPATH=src:experiments/e10_cfd \\
      .venvs/frost-mhd/bin/python experiments/e10_cfd/stats_table.py data/results/e10/stats3d_w5.json \\
      --converged "classical 128:5:none:0:t"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stats3d import scores_any


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("result", type=Path)
    ap.add_argument("--shells", type=int, default=12)
    ap.add_argument("--converged", default="", help="row name used as the second yardstick")
    args = ap.parse_args()
    d = json.loads(args.result.read_text())
    ref = np.asarray(d["dns_stats"])
    conv = np.asarray(d["rows"][args.converged]["mean_stats"]) if args.converged else None
    mhd = len(ref) > 30
    head = f"{'method':30s} {'ms':>7s} | spec DNS" + ("  spec conv" if conv is not None else "")
    head += " | enstr | skew err  flat err"
    if mhd:
        head += " || mag spec" + ("  mag conv" if conv is not None else "") + " | current | mag flat err | Hc err"
    print(head)
    for name, r in d["rows"].items():
        if "mean_stats" not in r:
            print(f"{name[:30]:30s} {r.get('ms_per_0.04', float('nan')):7.2f} | not finite")
            continue
        m = np.asarray(r["mean_stats"])
        a = scores_any(m, ref, 8, args.shells)
        c = scores_any(m, conv, 8, args.shells) if conv is not None else None
        line = f"{name[:30]:30s} {r['ms_per_0.04']:7.2f} | {a['spectrum_l1']:.4f}"
        line += f"    {c['spectrum_l1']:.4f}" if c else ""
        line += f" | {a['enstrophy_err']:+.3f} | {a['skewness_band_err']:+.3f}    {a['flatness_band_err']:+.3f}"
        if mhd:
            line += f"   || {a['mag_spectrum_l1']:.4f}" + (f"    {c['mag_spectrum_l1']:.4f}" if c else "")
            line += (f" | {a['mag_enstrophy_err']:+.3f}  | {a['mag_flatness_band_err']:+.3f}"
                     f"       | {a['cross_helicity_err']:+.4f}")
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
