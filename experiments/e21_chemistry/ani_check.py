"""Is the ANI-1xnr teacher kind sound? Energy and forces through the teacher server's own ``TorchANI`` class on the
methane-oxygen mixture: forces against central differences of the energy, and the time per call (7.4).

    PYTHONPATH=src:experiments/e14_baselines .venvs/frost-ani/bin/python experiments/e21_chemistry/ani_check.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    from teacher_server import TorchANI

    from frost.systems import E1_SYSTEMS

    at = E1_SYSTEMS["ch4_o2_270"]()
    model = TorchANI(str(ROOT / "third_party" / "ani-1xnr" / "model" / "ani-1xnr"), at.get_atomic_numbers(),
                     np.array(at.cell.array))
    x0 = at.get_positions()
    e0, f0 = model(x0)
    t0 = time.perf_counter()
    for _ in range(20):
        model(x0 + np.random.default_rng(0).normal(0, 0.01, x0.shape))
    per_call = (time.perf_counter() - t0) / 20
    h, rows = 1e-3, []
    for j, d in ((0, 0), (5, 1), (100, 2), (200, 0)):
        xp, xm = x0.copy(), x0.copy()
        xp[j, d] += h
        xm[j, d] -= h
        num = -(model(xp)[0] - model(xm)[0]) / (2 * h)
        rows.append({"atom": j, "axis": d, "fd": num, "analytic": float(f0[j, d])})
    err = max(abs(r["fd"] - r["analytic"]) for r in rows)
    print(f"ANI-1xnr ({model.members} ensemble members): E {e0:.3f} eV for {len(at)} atoms, |F| max {np.abs(f0).max():.3f} "
          f"eV/A, {1e3 * per_call:.2f} ms per call, max |analytic - fd| {err:.5f} eV/A", flush=True)
    for r in rows:
        print(f"   F[{r['atom']},{r['axis']}] analytic {r['analytic']:+.5f} fd {r['fd']:+.5f}", flush=True)
    out = ROOT / "data" / "results" / "e21"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ani_check.json").write_text(json.dumps({"energy_eV": e0, "ms_per_call": 1e3 * per_call, "members": model.members,
                                                   "max_force_diff_eV_A": err, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
