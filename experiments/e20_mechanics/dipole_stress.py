"""The teacher's stress on built dislocation cells (2026-11-18): is the cell stress-free before any dynamics?

The first screw-dipole builder absorbed the image sum's linear term into the cell tilts, which leaves ~1% homogeneous shear
strain; ``stress_free=True`` subtracts it and keeps the cell orthorhombic. One teacher call per cell (unrelaxed, so the
cores contribute a little) settles which construction is stress-free: the homogeneous part shows as shear stress of order
the shear modulus times the strain (~1.5-2 GPa for 1%).

    python experiments/e20_mechanics/dipole_stress.py w_screw3744 w_screw3744s [--gate w_screw3744s:0.3]

``--gate NAME:GPA`` exits 1 unless NAME's two shear stresses on the dislocation (xz and yz) are both under GPA, so a
pipeline queued after this check is blocked rather than run on a strained cell.
"""

from __future__ import annotations

import sys

import numpy as np


def main(argv: list[str]) -> int:
    from frost.systems import E1_SYSTEMS
    from frost.teachers import load_calc

    gates = dict(a.split("=", 1)[1].split(":") for a in argv if a.startswith("--gate="))
    if "--gate" in argv:
        i = argv.index("--gate")
        name, gpa = argv[i + 1].split(":")
        gates[name] = gpa
        argv = argv[:i] + argv[i + 2:]
    names = [a for a in argv if not a.startswith("--")]
    failed = False

    calc = load_calc("mp-medium", "cuda", "float32", enable_cueq=True)
    for name in names:
        at = E1_SYSTEMS[name]()
        at.calc = calc
        s = at.get_stress(voigt=True) * 160.21766  # eV/A^3 -> GPa; Voigt xx yy zz yz xz xy
        calc.reset()  # trap 3
        print(f"{name}: stress GPa xx {s[0]:+.3f} yy {s[1]:+.3f} zz {s[2]:+.3f} | yz {s[3]:+.3f} xz {s[4]:+.3f} "
              f"xy {s[5]:+.3f} | tilts {at.cell.array[0, 2]:.3f} {at.cell.array[1, 2]:.3f}", flush=True)
        if name in gates and max(abs(s[3]), abs(s[4])) >= float(gates[name]):
            print(f"{name}: shear above the gate {gates[name]} GPa", flush=True)
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
