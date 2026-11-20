"""Gate 3 probe: can the second message-passing layer be cached at all?

Why this is the deciding question. E2 measured that caching only layer 1 can
remove 22% of MACE's cost, a ceiling of about 1.27x, while caching both layers
removes 80%. So the useful speedup on MACE lives entirely in layer 2.

Layer 1 is easy because its edge tensor product depends only on the edge's own
geometry. Layer 2 is not: its first input is the node features its neighbours
computed in layer 1, and those change whenever anything nearby moves. A stored
geometric Jacobian dg/dr cannot see that dependence, so a layer-2 cache is
stale in two ways at once:

  * its VALUE ignores how the neighbours' features changed, and
  * its GRADIENT has no path back through those features, so any force that
    flows through that path is deleted -- the same kind of failure Theorem 6.1
    describes for a zeroth-order cache, one layer deeper.

The second effect would not shrink as displacement shrinks. So the sharp test
is the same as the week-0 probe: look at the error at displacement zero.

Configurations compared, all against exact evaluation:
  L1        layer 1 first-order cache, layer 2 exact  (the safe baseline)
  L1+L2geo  layer 2 first order in geometry only, neighbour state stale
            (what a stored geometric Jacobian can actually deliver)
  L1+L2full layer 2 full first order including the neighbour-state term
            (attribution check: it removes whatever L2geo loses)
  L1+L2zero layer 2 zeroth order (naive; the closure-saturation case)

Reading the result
  If L2geo tracks L1 closely at the displacements Gate 1 allows, layer 2 is
  cacheable with a geometric Jacobian and the 80% ceiling is reachable.
  If L2geo sits on a floor well above L1 even at zero displacement, the
  neighbour-state path carries real force, and caching layer 2 needs either a
  Jacobian with respect to node features (enormous) or recomputation wherever
  node features moved -- which is attenuated closure's job to bound.

Runs on CPU if the GPUs are busy.

Usage
  python experiments/e3_closure/probe_layer2.py --device cpu --frames 2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TRAJ = ROOT / "data" / "trajectories"
RESULTS = ROOT / "data" / "results" / "e3"

DISPLACEMENTS = [0.0, 1e-4, 1e-3, 5e-3, 0.01, 0.02, 0.03, 0.05]
CONFIGS = {
    "L1": ("first", "exact"),
    "L1+L2geo": ("first", "first_stale_a"),
    "L1+L2full": ("first", "first"),
    "L1+L2zero": ("first", "zeroth"),
}


def rms(x: np.ndarray) -> float:
    return float(np.sqrt((x**2).sum(1).mean()))


def main() -> int:
    from ase import Atoms
    from mace.calculators import mace_mp

    from frost.calibrate import _batch_for
    from frost.substitute import EdgeCacheSubstituter, forces
    from frost.trajectory import TrajectoryReader

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system", default="si512")
    ap.add_argument("--temp", type=int, default=300)
    ap.add_argument("--frames", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=24)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    calc = mace_mp(model="medium", device=args.device, default_dtype="float64")
    mods = dict(calc.models[0].named_modules())
    l1, l2 = mods["interactions.0.conv_tp"], mods["interactions.1.conv_tp"]

    # capture the layer-2 node-feature input, to report how much it moves
    box: dict = {}
    cap = l2.register_forward_hook(lambda m, i, o: box.__setitem__("a", i[0].detach()))

    rows = []
    with (
        TrajectoryReader(TRAJ / f"{args.system}_T{args.temp}.h5") as r,
        EdgeCacheSubstituter(l1) as s1,
        EdgeCacheSubstituter(l2) as s2,
    ):
        picks = np.linspace(0, r.n_steps - 1, args.frames).astype(int)
        for fi, step in enumerate(picks):
            at = Atoms(
                numbers=r.numbers,
                positions=np.asarray(r.positions[step]),
                cell=r.cell,
                pbc=True,
            )
            at.wrap()
            p0 = at.get_positions()
            rng = np.random.default_rng(2000 + fi)
            direction = rng.normal(size=p0.shape)
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
            keys_ref = _batch_for(calc, at, 6.0, args.device)[4]

            print(
                f"\nframe {fi + 1}/{args.frames} (step {step}), {len(keys_ref)} edges"
            )
            print(
                f"  {'dr (A)':>7} {'|da|/|a| p50':>13} {'p99':>8} | "
                + " ".join(f"{k:>10}" for k in CONFIGS)
                + "   (meV/A)"
            )
            for dr in DISPLACEMENTS:
                t0 = time.perf_counter()
                moved = at.copy()
                moved.set_positions(p0 + (dr / np.sqrt(2)) * direction)
                keys_cur = _batch_for(calc, moved, 6.0, args.device)[4]

                # reference: record both layers
                for s in (s1, s2):
                    s.ref_keys = s.cur_keys = None
                    s.mode = "record"
                forces(calc, at)
                a_ref = box["a"].clone()

                for s in (s1, s2):
                    s.ref_keys, s.cur_keys = keys_ref, keys_cur
                    s.mode = "exact"
                f_exact = forces(calc, moved)
                a_cur = box["a"]

                # neighbour-state change seen by layer 2, on edges present in both
                _, ir, ic = np.intersect1d(keys_ref, keys_cur, return_indices=True)
                da = (a_cur[ic] - a_ref[ir]).norm(dim=1)
                rel = (da / a_ref[ir].norm(dim=1).clamp_min(1e-12)).cpu().numpy()

                errs = {}
                for name, (m1, m2) in CONFIGS.items():
                    s1.mode, s2.mode = m1, m2
                    errs[name] = rms(forces(calc, moved) - f_exact) * 1e3
                row = {
                    "system": args.system,
                    "temperature": args.temp,
                    "frame": int(step),
                    "displacement_A": dr,
                    "da_rel_p50": float(np.median(rel)),
                    "da_rel_p99": float(np.percentile(rel, 99)),
                    "exact_force_rms_meV_per_A": rms(f_exact) * 1e3,
                    **{f"err_{k}_meV_per_A": v for k, v in errs.items()},
                }
                rows.append(row)
                print(
                    f"  {dr:7.4f} {row['da_rel_p50']:13.2e} {row['da_rel_p99']:8.2e} | "
                    + " ".join(f"{errs[k]:10.4f}" for k in CONFIGS)
                    + f"   [{time.perf_counter() - t0:.0f}s]",
                    flush=True,
                )
    cap.remove()

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"layer2_{args.system}.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\n{len(rows)} rows -> {out}")

    # verdict at zero displacement: is there a floor?
    zero = [r for r in rows if r["displacement_A"] == 0.0]
    if zero:
        floor = float(np.median([r["err_L1+L2geo_meV_per_A"] for r in zero]))
        base = float(np.median([r["err_L1_meV_per_A"] for r in zero]))
        f_rms = float(np.median([r["exact_force_rms_meV_per_A"] for r in zero]))
        print("\n" + "=" * 64)
        print(
            f"at zero displacement: L1 error {base:.4f}, L1+L2geo error {floor:.4f} meV/A "
            f"({floor / f_rms:.1%} of the force)"
        )
        if floor > 1.0:
            print("  -> a floor that does not vanish: caching layer 2 with a geometric")
            print("     Jacobian deletes the force path through neighbour features.")
        else:
            print("  -> no floor: the neighbour-feature path carries negligible force.")
        print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
