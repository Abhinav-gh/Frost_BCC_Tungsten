"""E1 step 1: log NVT trajectories for the offline cache replay.

Research question (RQ1). How much temporal redundancy exists at edge
granularity, across systems, temperatures and phases, and how does it depend on
the tolerance?

This script only produces the trajectories. `simulate_cache.py` replays them
and `analyse.py` turns the replay into the phi(eps) curves and Figure 1.

Protocol (Frost.pdf appendix H.1):
  1. Equilibrate 20 ps NVT Langevin at 300 K and 1000 K, discard.
  2. Run 10,000 production steps at dt = 1 fs, logging every step.

Usage
  python experiments/e1_redundancy/run_trajectory.py --system si512 --temp 300
  python experiments/e1_redundancy/run_trajectory.py --all --steps 10000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from frost.systems import E1_SYSTEMS
from frost.trajectory import run_md

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "trajectories"

# The systems E1 names, in the order the plan runs them: predicted-favourable
# first, so an unfavourable pilot stops the sweep early.
ORDER = ["si512", "silica1152", "li3po4", "cu500", "water216"]
TEMPS = [300, 1000]


def build_calculator(model: str, device: str, dtype: str, teacher: str = "mp-medium"):
    if teacher != "mp-medium":
        from frost.teachers import load_calc

        return load_calc(teacher, device, dtype, enable_cueq=False)
    from mace.calculators import mace_mp

    return mace_mp(model=model, device=device, default_dtype=dtype)


def one(
    system: str,
    temp: int,
    *,
    steps: int,
    equil: int,
    model: str,
    device: str,
    dtype: str,
    cutoff: float,
    skin: float,
    overwrite: bool,
    teacher: str = "mp-medium",
    label: str = "",
) -> dict:
    out = OUT / f"{label or system}_T{temp}.h5"
    if out.exists() and not overwrite:
        print(f"[skip] {out.name} exists")
        return {"system": system, "temp": temp, "path": str(out), "skipped": True}

    atoms = E1_SYSTEMS[system]()
    print(f"\n=== {system} @ {temp} K : {len(atoms)} atoms, {steps} steps ===")
    calc = build_calculator(model, device, dtype, teacher)
    t0 = time.perf_counter()
    run_md(
        atoms,
        calc,
        steps=steps,
        timestep_fs=1.0,
        temperature_K=temp,
        out=out,
        cutoff=cutoff,
        skin=skin,
        equilibrate_steps=equil,
        extra_attrs={"teacher": teacher},
    )
    return {
        "system": system,
        "temp": temp,
        "n_atoms": len(atoms),
        "steps": steps,
        "equil": equil,
        "model": model,
        "dtype": dtype,
        "cutoff": cutoff,
        "skin": skin,
        "wall_s": round(time.perf_counter() - t0, 1),
        "path": str(out.relative_to(ROOT)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--system", choices=list(E1_SYSTEMS), help="one system")
    p.add_argument("--temp", type=int, choices=TEMPS)
    p.add_argument("--all", action="store_true", help="every system x temperature")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--equil", type=int, default=20000, help="20 ps at 1 fs")
    p.add_argument("--model", default="medium")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--cutoff", type=float, default=6.0, help="MACE-MP-0 r_c")
    p.add_argument("--skin", type=float, default=1.0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--teacher", default="mp-medium", help="frost.teachers name, recorded in the file")
    p.add_argument("--label", default="", help="run name instead of the system's (e.g. water216off)")
    args = p.parse_args()

    if args.all:
        jobs = [(s, t) for s in ORDER for t in TEMPS]
    elif args.system:
        jobs = [(args.system, t) for t in ([args.temp] if args.temp else TEMPS)]
    else:
        p.error("pass --system or --all")

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for s, t in jobs:
        try:
            manifest.append(
                one(
                    s,
                    t,
                    steps=args.steps,
                    equil=args.equil,
                    model=args.model,
                    device=args.device,
                    dtype=args.dtype,
                    cutoff=args.cutoff,
                    skin=args.skin,
                    overwrite=args.overwrite,
                    teacher=args.teacher,
                    label=args.label,
                )
            )
        except Exception as exc:
            print(f"[FAIL] {s} @ {t}K: {type(exc).__name__}: {exc}")
            manifest.append({"system": s, "temp": t, "error": repr(exc)})

    mpath = OUT / "manifest.json"
    old = json.loads(mpath.read_text()) if mpath.exists() else []
    mpath.write_text(json.dumps(old + manifest, indent=2))
    print(f"\nmanifest -> {mpath}")
    ok = sum("error" not in m for m in manifest)
    print(f"{ok}/{len(manifest)} trajectories written")
    return 0 if ok == len(manifest) else 1


if __name__ == "__main__":
    raise SystemExit(main())
