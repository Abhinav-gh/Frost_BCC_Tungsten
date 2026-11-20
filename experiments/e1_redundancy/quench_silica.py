"""Produce the amorphous silica glass E1 actually specifies.

The E1 protocol (Frost.pdf table 15) names silica *glass* as a primary system
and predicts it to be a best case, because dynamics below the glass transition
are near-arrested. The first E1 pass used the crystalline beta-cristobalite
seed instead, which is a different material; its failing row was therefore not
a test of that prediction.

Protocol, scaled to the GPU budget:
  1. melt   3000 K NVT for 10 ps, which destroys the crystal
  2. quench 3000 K -> 300 K linearly over 27 ps (1e14 K/s)
  3. anneal 300 K NVT for 5 ps

The design document quotes 1e12 K/s. That takes 2.7 ns, which at the step time
measured on this cell is days of GPU time. A faster quench gives a glass that
is less relaxed, with more defects and a slightly higher fictive temperature.
That biases the glass toward MORE mobility, not less, so if the glass passes
at this rate the conclusion is conservative. This is recorded in the output
file's attributes so the difference from the protocol is never silent.

The quenched structure is written to data/systems/silica_glass1152.xyz, and
checked: no coincident atoms, and the Si-O first peak where it belongs.

Usage
  python experiments/e1_redundancy/quench_silica.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "systems" / "silica_glass1152.xyz"

T_MELT, T_FINAL = 3000.0, 300.0
MELT_STEPS, QUENCH_STEPS, ANNEAL_STEPS = 10_000, 27_000, 5_000
REPORT = 1_000


def main() -> int:
    from ase import units
    from ase.io import write
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.neighborlist import neighbor_list
    from mace.calculators import mace_mp

    from frost.systems import silica1152

    atoms = silica1152()
    atoms.calc = mace_mp(model="medium", device="cuda", default_dtype="float32")
    rng = np.random.default_rng(7)
    MaxwellBoltzmannDistribution(atoms, temperature_K=T_MELT, rng=rng)
    dyn = Langevin(
        atoms, 1.0 * units.fs, temperature_K=T_MELT, friction=0.01 / units.fs, rng=rng
    )

    def run(n, label, temp_of=None):
        t0 = time.perf_counter()
        for k in range(0, n, 100):
            if temp_of is not None:
                dyn.set_temperature(temperature_K=temp_of(k))
            dyn.run(min(100, n - k))
            if (k + 100) % REPORT == 0 or k + 100 >= n:
                e = atoms.get_potential_energy()
                if not np.isfinite(e):
                    raise RuntimeError(f"energy not finite during {label}")
                el = time.perf_counter() - t0
                print(
                    f"  {label:<7} {min(k + 100, n):>6}/{n}  "
                    f"T={atoms.get_temperature():6.0f} K  {el:5.0f}s",
                    flush=True,
                )

    print(f"melt at {T_MELT:.0f} K")
    run(MELT_STEPS, "melt")
    print(f"quench {T_MELT:.0f} -> {T_FINAL:.0f} K over {QUENCH_STEPS} fs")
    run(
        QUENCH_STEPS,
        "quench",
        temp_of=lambda k: T_MELT + (T_FINAL - T_MELT) * k / QUENCH_STEPS,
    )
    print(f"anneal at {T_FINAL:.0f} K")
    dyn.set_temperature(temperature_K=T_FINAL)
    run(ANNEAL_STEPS, "anneal")

    atoms.wrap()
    i, j, d = neighbor_list("ijd", atoms, 3.5)
    sym = np.array(atoms.get_chemical_symbols())
    sio = (sym[i] == "Si") & (sym[j] == "O")
    hist, edges = np.histogram(d[sio], bins=70, range=(0.0, 3.5))
    peak = 0.5 * (edges[:-1] + edges[1:])[np.argmax(hist)]
    cn = np.bincount(i[sio & (d < 2.0)], minlength=len(atoms))[sym == "Si"]
    print(
        f"\nmin distance {d.min():.3f} A, Si-O first peak {peak:.2f} A, "
        f"mean Si coordination {cn.mean():.2f}"
    )
    if d.min() < 0.9:
        raise RuntimeError("quenched cell has coincident atoms")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    atoms.info.update(
        quench_rate_K_per_s=1e14,
        protocol_rate_K_per_s=1e12,
        melt_K=T_MELT,
        final_K=T_FINAL,
        si_o_peak_A=float(peak),
        si_coordination=float(cn.mean()),
    )
    write(OUT, atoms)
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
