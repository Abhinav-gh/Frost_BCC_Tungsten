"""Time GPU4PySCF's energy and analytic forces (the ``pyscf:`` teacher kind) on small molecules, the first call and scanner
calls after a small move, and check the forces against a finite difference (further_expansions_beyond.md 7.1).

    PYTHONPATH=.tools/pyscf-overlay:experiments/e14_baselines .venvs/frost-pyscf/bin/python experiments/e18_dft/dft_timing.py

The overlay supplies the unversioned libcusolver.so gpu4pyscf searches for (scripts/pyscf_overlay.sh). The DFT class is
the teacher server's own (``teacher_server.GPUDFT``), so this times exactly what a run will call.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import molecule

ROOT = Path(__file__).resolve().parents[2]


def water_cluster(n: int = 8, seed: int = 0) -> Atoms:
    """n waters on a 2.9 A grid with random orientations (a starting geometry for timing, not an equilibrium one)."""
    rng = np.random.default_rng(seed)
    out, side = Atoms(), int(np.ceil(n ** (1 / 3)))
    for i in range(n):
        w = molecule("H2O")
        w.rotate(rng.uniform(0, 360), rng.normal(size=3), center=(0, 0, 0))
        w.translate(2.9 * np.array([i % side, (i // side) % side, i // side ** 2], dtype=float))
        out += w
    return out


def main() -> int:
    import cupy
    from teacher_server import GPUDFT  # experiments/e14_baselines on PYTHONPATH

    print("GPU:", cupy.cuda.runtime.getDeviceProperties(0)["name"].decode(), flush=True)
    rows = []
    for name, at in (("ethanol", molecule("CH3CH2OH")), ("water8", water_cluster(8))):
        for spec in ("pbe/def2-svp/df", "b3lyp/def2-svp/df", "pbe/def2-tzvp/df"):
            dft = GPUDFT(spec, at.get_atomic_numbers())
            x = at.get_positions()
            t0 = time.perf_counter()
            e, f = dft(x)
            first = time.perf_counter() - t0
            rng = np.random.default_rng(0)
            times = []
            for _ in range(5):
                x = x + rng.normal(0, 0.005, x.shape)  # an MD-sized move: the scanner restarts from the last density
                t0 = time.perf_counter()
                dft(x)
                times.append(time.perf_counter() - t0)
            rows.append({"molecule": name, "atoms": len(at), "spec": spec, "energy_eV": e, "first_s": first,
                         "next_s": float(np.mean(times)), "fmax_eV_A": float(np.abs(f).max())})
            print(f"{name:8s} {len(at):3d} atoms {spec:18s} E {e:12.4f} eV  first {first:6.2f} s  next "
                  f"{np.mean(times):6.3f} s", flush=True)
    # forces against a central finite difference of the energy (PBE/def2-SVP/DF, ethanol, three coordinates)
    at = molecule("CH3CH2OH")
    dft = GPUDFT("pbe/def2-svp/df", at.get_atomic_numbers())
    x0 = at.get_positions()
    _, f0 = dft(x0)
    h, checks = 1e-3, []
    for j, d in ((2, 0), (0, 1), (6, 2)):
        xp, xm = x0.copy(), x0.copy()
        xp[j, d] += h
        xm[j, d] -= h
        num = -(dft(xp)[0] - dft(xm)[0]) / (2 * h)
        checks.append({"atom": j, "axis": d, "finite_difference": num, "analytic": float(f0[j, d])})
        print(f"F[{j},{d}]: finite difference {num:.5f}, analytic {f0[j, d]:.5f} eV/A", flush=True)
    out = ROOT / "data" / "results" / "e18"
    out.mkdir(parents=True, exist_ok=True)
    (out / "dft_timing.json").write_text(json.dumps({"rows": rows, "force_checks": checks}, indent=1))
    print("->", out / "dft_timing.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
