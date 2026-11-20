"""E1: the Gate 1 measurement, and Figure 1.

The gate asks one question: is there a geometric threshold at which most edges
stay cached AND the induced force error is below the model's own noise floor?

Answering it needs two curves, measured independently and then crossed:

  1. **phi(dr)** -- the dirty fraction produced by thresholding edge
     displacement at ``dr``. Pure geometry, read off the logged trajectories,
     no model and no surrogate.
  2. **force error(dr)** -- what that same threshold actually costs, measured by
     substituting a first-order cache into the model on sampled frames and
     comparing forces against exact evaluation.

Crossing them gives phi at the largest dr whose force error stays under the
5 meV/A ceiling. Doing it this way keeps the gate independent of how well the
sensitivity surrogate is fitted; the surrogate is a runtime convenience, not
evidence.

Falsification (Frost.pdf 12.7.1). Stop if phi > 0.5 at every dr whose induced
force error is below 5 meV/A, on the systems predicted favourable.

Usage
  python experiments/e1_redundancy/analyse.py --force-error   # curve 2, needs GPU
  python experiments/e1_redundancy/analyse.py --phi           # curve 1, CPU only
  python experiments/e1_redundancy/analyse.py --gate          # cross them
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from frost.cache_sim import replay
from frost.predicate import SensitivityTable
from frost.trajectory import TrajectoryReader

ROOT = Path(__file__).resolve().parents[2]
TRAJ = ROOT / "data" / "trajectories"
RESULTS = ROOT / "data" / "results" / "e1"

# Geometric thresholds, spanning from well under one step of thermal motion
# (6.5e-3 A on Si at 300 K) up to the vibrational amplitude where displacement
# saturates (about 0.27 A).
THRESHOLDS = [0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 0.30]

FAVOURABLE = {"si512", "silica1152", "silica_glass1152"}
FORCE_ERROR_CEILING_meV = 5.0
PHI_CEILING = 0.5


# --------------------------------------------------------------------------
# curve 1: phi(dr), geometry only
# --------------------------------------------------------------------------
def phi_curves(max_steps: int | None, stride: int, only: str = "*") -> list[dict]:
    dummy = SensitivityTable.constant([1], cutoff=6.0, value=1.0)
    rows = []
    for path in sorted(TRAJ.glob(f"{only}.h5")):
        system = path.stem.rsplit("_T", 1)[0]
        print(f"\n=== {path.stem} ===")
        with TrajectoryReader(path) as r:
            temp = float(r.attrs.get("temperature_K", 0))
            for dr in THRESHOLDS:
                res = replay(
                    r,
                    dummy,
                    eps_meV_per_A=0.0,
                    order=1,
                    max_steps=max_steps,
                    stride=stride,
                    geometric_threshold=dr,
                )
                row = res.summary()
                row.update(
                    system=system,
                    temperature=temp,
                    threshold_A=dr,
                    trajectory=path.name,
                )
                rows.append(row)
                print(
                    f"  dr <= {dr:6.3f} A  phi = {row['phi']:.4f}  "
                    f"CV(tau) = {row['cv_tau']:.3f}  "
                    f"mean refresh interval = {row['mean_refresh_interval']:.1f} steps"
                )
    return rows


# --------------------------------------------------------------------------
# curve 2: induced force error at a given threshold
# --------------------------------------------------------------------------
def force_error_curve(
    system: str, temp: int, n_frames: int, model: str, device: str
) -> list[dict]:
    """Substitute a first-order cache and measure the force error it costs.

    Frames are drawn from the logged trajectory, so this is the error on real
    configurations rather than on a rattled crystal. Each atom is displaced by
    dr/sqrt(2) in a random direction, which makes the typical EDGE move by
    about dr -- the quantity the predicate thresholds on.
    """
    from ase import Atoms
    from mace.calculators import mace_mp

    from frost.calibrate import _batch_for
    from frost.substitute import EdgeCacheSubstituter, force_error_at

    path = TRAJ / f"{system}_T{temp}.h5"
    calc = mace_mp(model=model, device=device, default_dtype="float64")
    module = dict(calc.models[0].named_modules())["interactions.0.conv_tp"]

    rows: list[dict] = []
    skipped = 0
    with TrajectoryReader(path) as r, EdgeCacheSubstituter(module) as sub:
        picks = np.linspace(0, r.n_steps - 1, n_frames).astype(int)
        for n, step in enumerate(picks):
            at = Atoms(
                numbers=r.numbers,
                positions=np.asarray(r.positions[step]),
                cell=r.cell,
                pbc=True,
            )
            at.wrap()
            rng = np.random.default_rng(1000 + n)
            direction = rng.normal(size=at.positions.shape)
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
            pos0 = at.get_positions()

            keys_ref = _batch_for(calc, at, 6.0, device)[4]
            for dr in THRESHOLDS:
                moved = at.copy()
                moved.set_positions(pos0 + (dr / np.sqrt(2)) * direction)
                keys_cur = _batch_for(calc, moved, 6.0, device)[4]
                matched = len(np.intersect1d(keys_ref, keys_cur))
                err, rms = force_error_at(
                    calc,
                    sub,
                    at,
                    moved,
                    mode="first",
                    ref_keys=keys_ref,
                    cur_keys=keys_cur,
                )
                rows.append(
                    {
                        "system": system,
                        "temperature": float(temp),
                        "frame": int(step),
                        "threshold_A": dr,
                        "force_error_meV_per_A": err * 1e3,
                        "exact_force_rms_meV_per_A": rms * 1e3,
                        "edges_cached": int(matched),
                        "edges_total": len(keys_cur),
                    }
                )
            print(f"  frame {n + 1}/{n_frames} (step {step})", flush=True)
    if skipped:
        print(
            f"  {skipped} points skipped on a neighbour-list rebuild "
            "(the edge sets are not comparable there)"
        )
    return rows


# --------------------------------------------------------------------------
# cross the two curves
# --------------------------------------------------------------------------
def gate(
    phi_rows: list[dict],
    err_by_system: dict[str, list[dict]],
    ceiling_meV: float = FORCE_ERROR_CEILING_meV,
) -> tuple[bool, str]:
    """Apply Gate 1, giving each system its own threshold.

    The tolerance is a per-run knob, so the criterion asks whether each
    favourable system has SOME threshold with force error under the ceiling
    and phi under 0.5 there. An earlier version forced one shared threshold
    across systems, which pinned silicon to silica's much tighter value and
    reported phi near 0.85 for a system that actually passes.

    Force error is taken as the maximum over frames and interpolated linearly
    in displacement, which the measured curves justify: first-order error is
    linear in dr to within a few percent over this range.
    """
    if not err_by_system:
        return False, "no force-error curves; run --force-error first"

    phi: dict[tuple, dict[float, float]] = {}
    for row in phi_rows:
        phi.setdefault((row["system"], float(row["temperature"])), {})[
            row["threshold_A"]
        ] = row["phi"]

    lines = [
        f"ceiling {ceiling_meV:g} meV/A on induced force error, phi < {PHI_CEILING}"
    ]
    favourable_pass = []
    favourable_seen = []
    for system, rows in sorted(err_by_system.items()):
        by: dict[float, float] = {}
        for r in rows:
            by[r["threshold_A"]] = max(
                by.get(r["threshold_A"], 0.0), r["force_error_meV_per_A"]
            )
        d = np.array(sorted(by))
        e = np.array([by[x] for x in d])
        if e[0] > ceiling_meV:
            lines.append(f"  {system}: even the smallest threshold exceeds the ceiling")
            continue
        dr_max = float(np.interp(ceiling_meV, e, d))
        slope = float(np.polyfit(d, e, 1)[0])
        lines.append(
            f"  {system}: force error slope {slope:.0f} meV/A per A, "
            f"dr_max = {dr_max:.4f} A"
        )
        for (sysname, temp), curve in sorted(phi.items()):
            if sysname != system:
                continue
            grid = sorted(curve)
            p = float(np.interp(dr_max, grid, [curve[g] for g in grid]))
            ok = p < PHI_CEILING
            lines.append(
                f"      {temp:>6.0f} K   phi = {p:.3f}   {'PASS' if ok else 'fail'}"
            )
            if system in FAVOURABLE and temp == 300.0:
                favourable_seen.append(system)
                if ok:
                    favourable_pass.append(system)

    # A favourable system with a phi curve but no force-error curve must not
    # vanish from the verdict: that is how the glass once went missing after its
    # force-error step ran out of GPU memory.
    missing = sorted({k[0] for k in phi if k[0] in FAVOURABLE} - set(err_by_system))
    for m in missing:
        lines.append(f"  {m}: phi measured but NO force-error curve -- verdict incomplete")

    ok = bool(favourable_pass)
    incomplete = f" (INCOMPLETE: {', '.join(missing)} missing)" if missing else ""
    head = (
        f"GATE 1 {'PASSES' if ok else 'FAILS'}{incomplete}: "
        f"{len(favourable_pass)}/{len(favourable_seen)} favourable systems pass at 300 K"
    )
    return ok, head + "\n" + "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phi", action="store_true")
    p.add_argument("--force-error", action="store_true")
    p.add_argument("--gate", action="store_true")
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--frames", type=int, default=6)
    p.add_argument("--system", default="si512")
    p.add_argument("--temp", type=int, default=300)
    p.add_argument("--ceiling", type=float, default=FORCE_ERROR_CEILING_meV)
    p.add_argument("--only", default="*", help="glob over trajectory stems for --phi")
    p.add_argument("--model", default="medium")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    phi_path = RESULTS / "phi_vs_threshold.json"
    err_path = RESULTS / f"ferr_{args.system}.json"

    if args.phi:
        rows = phi_curves(args.max_steps, args.stride, args.only)
        if args.only != "*" and phi_path.exists():
            keep = {r["trajectory"] for r in rows}
            rows = [
                r
                for r in json.loads(phi_path.read_text())
                if r["trajectory"] not in keep
            ] + rows
        phi_path.write_text(json.dumps(rows, indent=2))
        print(f"\n{len(rows)} rows -> {phi_path}")

    if args.force_error:
        rows = force_error_curve(
            args.system, args.temp, args.frames, args.model, args.device
        )
        err_path.write_text(json.dumps(rows, indent=2))
        print(f"\n{len(rows)} rows -> {err_path}")

    if args.gate:
        phi_rows = json.loads(phi_path.read_text()) if phi_path.exists() else []
        err_by_system = {
            f.stem.removeprefix("ferr_"): json.loads(f.read_text())
            for f in sorted(RESULTS.glob("ferr_*.json"))
        }
        ok, text = gate(phi_rows, err_by_system, args.ceiling)
        print("\n" + "=" * 64)
        print(text)
        print("=" * 64)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
