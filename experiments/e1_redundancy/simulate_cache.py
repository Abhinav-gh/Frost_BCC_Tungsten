"""E1 step 2: replay logged trajectories under a simulated per-edge cache.

Research question (RQ1). How much temporal redundancy exists at edge
granularity, and how does the dirty fraction depend on the tolerance?

Success / falsification (Frost.pdf section 12.7.1, Gate 1)
    PROCEED if phi < 0.5 at an eps whose induced force error stays below
    5 meV/A, on the systems predicted favourable (silica, Si), for EITHER
    cache order.
    STOP otherwise, and write the characterisation paper instead.

Break-even dirty fractions, read from the speedup surface (Frost.pdf table 5):

    theta      0.35   0.50   0.70   0.85
    phi*       0.11   0.42   0.63   0.73

so Allegro (theta ~ 0.85) tolerates phi up to 0.73 and MACE layer-1-only
(theta ~ 0.5) needs phi below roughly 0.42.

Usage
  python experiments/e1_redundancy/simulate_cache.py --calibrate
  python experiments/e1_redundancy/simulate_cache.py --all
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from frost.cache_sim import fixed_stride, replay
from frost.predicate import SensitivityTable
from frost.trajectory import TrajectoryReader

ROOT = Path(__file__).resolve().parents[2]
TRAJ = ROOT / "data" / "trajectories"
RESULTS = ROOT / "data" / "results" / "e1"
TABLES = ROOT / "data" / "calibration"

# Eight tolerances spanning four decades, as the protocol specifies.
EPS_SWEEP = [0.1, 0.3, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
STRIDES = [2, 5, 10, 25, 50, 100]

# Break-even phi by cacheable share theta (Frost.pdf table 5 / appendix N.2).
BREAK_EVEN = {0.35: 0.11, 0.50: 0.42, 0.70: 0.63, 0.85: 0.73}


def calibrate(
    system: str, n_frames: int, site: str, cutoff: float, model: str, device: str
) -> tuple[Path, Path]:
    """Fit S_hat for both cache orders from sampled frames of one system."""
    from mace.calculators import mace_mp

    from frost.calibrate import collect_samples, fit_tables
    from frost.systems import E1_SYSTEMS

    TABLES.mkdir(parents=True, exist_ok=True)
    atoms0 = E1_SYSTEMS[system]()
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(n_frames):
        a = atoms0.copy()
        a.set_positions(a.get_positions() + rng.normal(0, 0.05, a.positions.shape))
        frames.append(a)

    calc = mace_mp(model=model, device=device, default_dtype="float64")
    print(f"  sampling {len(frames)} frames of {system} ...")
    t0 = time.perf_counter()
    samples = collect_samples(calc, frames, site=site, cutoff=cutoff)
    print(
        f"  {len(samples['r'])} edge samples in {time.perf_counter() - t0:.0f}s "
        f"({samples['matched_fraction']:.1%} of rows matched across perturbation)"
    )

    s0, s1 = fit_tables(
        samples, species=np.unique(atoms0.get_atomic_numbers()), cutoff=cutoff
    )
    p0 = TABLES / f"{system}_order0.npz"
    p1 = TABLES / f"{system}_order1.npz"
    s0.save(p0)
    s1.save(p1)
    nz0 = float(np.count_nonzero(s0.table)) / s0.table.size
    nz1 = float(np.count_nonzero(s1.table)) / s1.table.size
    print(f"  order 0: max S_hat = {s0.table.max():.4g}, {nz0:.0%} of cells filled")
    print(f"  order 1: max S_hat = {s1.table.max():.4g}, {nz1:.0%} of cells filled")
    print(f"  -> {p0.name}, {p1.name}")
    return p0, p1


def sweep_one(
    path: Path,
    tables: dict[int, SensitivityTable],
    *,
    max_steps: int | None,
    gamma: str,
) -> list[dict]:
    rows = []
    with TrajectoryReader(path) as r:
        system = path.stem.rsplit("_T", 1)[0]
        temp = float(r.attrs.get("temperature_K", 0))
        for order, table in sorted(tables.items()):
            for eps in EPS_SWEEP:
                res = replay(
                    r,
                    table,
                    eps_meV_per_A=eps,
                    order=order,
                    gamma=gamma,
                    max_steps=max_steps,
                )
                row = res.summary()
                row.update(system=system, temperature=temp, trajectory=path.name)
                rows.append(row)
                print(
                    f"    order={order} eps={eps:>5.1f} meV/A  "
                    f"phi={row['phi']:.4f}  CV(tau)={row['cv_tau']:.3f}"
                )
        for k in STRIDES:
            res = fixed_stride(r, k=k, max_steps=max_steps)
            row = res.summary()
            row.update(
                system=system,
                temperature=temp,
                trajectory=path.name,
                baseline=f"stride-{k}",
                stride=k,
            )
            rows.append(row)
    return rows


def gate_verdict(rows: list[dict]) -> tuple[bool, str]:
    """Apply the Gate 1 criterion to the sweep."""
    favourable = {"si512", "silica1152"}
    best = {}
    for row in rows:
        if row.get("baseline") or row["system"] not in favourable:
            continue
        # 5 meV/A is the induced-force-error ceiling the gate allows.
        if row["eps_meV_per_A"] > 5.0:
            continue
        key = (row["system"], row["temperature"], row["order"])
        if key not in best or row["phi"] < best[key]["phi"]:
            best[key] = row
    if not best:
        return False, "no rows on the favourable systems within the 5 meV/A ceiling"
    passing = [v for v in best.values() if v["phi"] < 0.5]
    lines = [
        f"    {k[0]} @ {k[1]:.0f}K order={k[2]}: best phi = {v['phi']:.4f}"
        for k, v in sorted(best.items())
    ]
    ok = bool(passing)
    verdict = ("GATE 1 PASSES" if ok else "GATE 1 FAILS") + "\n" + "\n".join(lines)
    return ok, verdict


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibrate", action="store_true", help="fit S_hat first")
    p.add_argument("--calib-system", default="si512")
    p.add_argument("--calib-frames", type=int, default=32)
    p.add_argument("--site", default="interactions.0.conv_tp")
    p.add_argument("--cutoff", type=float, default=6.0)
    p.add_argument("--model", default="medium")
    p.add_argument("--device", default="cuda")
    p.add_argument("--all", action="store_true", help="sweep every trajectory")
    p.add_argument("--trajectory", type=Path)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--gamma", default="independent")
    args = p.parse_args()

    if args.calibrate:
        calibrate(
            args.calib_system,
            args.calib_frames,
            args.site,
            args.cutoff,
            args.model,
            args.device,
        )

    if not (args.all or args.trajectory):
        return 0

    tables = {}
    for order in (0, 1):
        tp = TABLES / f"{args.calib_system}_order{order}.npz"
        if tp.exists():
            tables[order] = SensitivityTable.load(tp)
    if not tables:
        print("no sensitivity tables; run with --calibrate first")
        return 2

    paths = [args.trajectory] if args.trajectory else sorted(TRAJ.glob("*.h5"))
    if not paths:
        print(f"no trajectories in {TRAJ}")
        return 2

    RESULTS.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for path in paths:
        print(f"\n=== {path.name} ===")
        try:
            rows += sweep_one(path, tables, max_steps=args.max_steps, gamma=args.gamma)
        except Exception as exc:
            print(f"  [FAIL] {type(exc).__name__}: {exc}")

    out = RESULTS / "sweep.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\n{len(rows)} rows -> {out}")

    ok, verdict = gate_verdict(rows)
    print("\n" + "=" * 60)
    print(verdict)
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
