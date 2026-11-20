"""Are the DFT teacher's forces the derivative of its energy? (further_expansions_beyond.md 7.1)

An NVE run conserves energy only if the forces are -dE/dx of the energy it reports. GPU4PySCF's default gradient omits
the response of the moving quadrature grid, and on ethanol it disagreed with a finite difference by 0.004 eV/A on one
component (2026-11-14). This compares analytic forces with central differences of the energy on several coordinates,
with the grid response off and on (``teacher_server.GPUDFT``'s ``/nogridresp`` option), and times both.

    PYTHONPATH=.tools/pyscf-overlay:experiments/e14_baselines .venvs/frost-pyscf/bin/python experiments/e18_dft/dft_consistency.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from ase.build import molecule

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    from teacher_server import GPUDFT  # experiments/e14_baselines on PYTHONPATH

    at = molecule("CH3CH2OH")
    rng = np.random.default_rng(1)
    x0 = at.get_positions() + rng.normal(0, 0.03, (len(at), 3))  # off the symmetric geometry: no zero components
    coords = [(0, 0), (2, 1), (4, 2), (6, 0), (8, 1)]
    h = 2e-3
    out = {}
    for label, spec in (("no grid response", "pbe/def2-svp/df/nogridresp"), ("grid response", "pbe/def2-svp/df")):
        dft = GPUDFT(spec, at.get_atomic_numbers())
        _, f0 = dft(x0)
        t0 = time.perf_counter()
        for _ in range(3):
            dft(x0 + rng.normal(0, 0.003, x0.shape))
        per_call = (time.perf_counter() - t0) / 3
        rows = []
        for j, d in coords:
            xp, xm = x0.copy(), x0.copy()
            xp[j, d] += h
            xm[j, d] -= h
            num = -(dft(xp)[0] - dft(xm)[0]) / (2 * h)
            rows.append({"atom": j, "axis": d, "finite_difference": num, "analytic": float(f0[j, d]),
                         "diff": float(f0[j, d] - num)})
        err = max(abs(r["diff"]) for r in rows)
        out[label] = {"rows": rows, "max_abs_diff_eV_A": err, "seconds_per_call": per_call}
        print(f"{label:17s} max |analytic - finite difference| {err:.5f} eV/A over {len(rows)} components, "
              f"{per_call:.3f} s per call", flush=True)
        for r in rows:
            print(f"   F[{r['atom']},{r['axis']}] analytic {r['analytic']:+.5f} fd {r['finite_difference']:+.5f} "
                  f"diff {r['diff']:+.5f}", flush=True)
    res = ROOT / "data" / "results" / "e18"
    res.mkdir(parents=True, exist_ok=True)
    (res / "dft_consistency.json").write_text(json.dumps(out, indent=1))
    print("->", res / "dft_consistency.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
