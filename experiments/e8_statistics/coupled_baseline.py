"""E8: the teacher's own timestep bias, measured at the precision Frost is held to.

The 4 fs Langevin baseline was admitted on unpaired arms (+/- 0.057 meV/atom),
while Frost's residual bias is now measured on seed-paired arms (+/- 0.007).
Holding one to a standard ten times finer than the other cannot rank them, so this
measures the baseline's own bias the same way.

Two teachers, dt = 1 fs and dt = 4 fs, walk **one** Brownian path. BAOAB's only
noise is its O step, an Ornstein-Uhlenbeck update v <- c v + sigma sqrt(1 - c^2) xi
with c = exp(-gamma dt). Four fine steps compose to

    v4 = c^4 v0 + sigma sqrt(1 - c^2) sum_i c^(3-i) xi_i,

and one coarse step over the same interval is v4 = C v0 + sigma sqrt(1 - C^2) xi_c
with C = c^4, so

    xi_c = sqrt((1 - c^2) / (1 - C^2)) sum_i c^(3-i) xi_i

is exactly N(0,1) and built from the same four fine draws. The coupling is exact,
not approximate: at gamma = 0.05/fs the weights are (0.8607, 0.9048, 0.9512, 1)
and (1 - c^2) sum w^2 matches 1 - C^2 to eight figures, which ``--check`` asserts
and which the runner refuses to proceed without.

The pairing is a variance-reduction device, not a proof. The two arms see the same
noise but different forces, so they separate over ~tau and the reduction decays.
The paired difference is unbiased regardless; what decays is its precision, so the
report gives the paired SE as a function of trajectory length rather than one
number, which also shows how long the coupling is worth keeping.

Output: data/results/e8/coupled_<run>_<tag>.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TRAJ = ROOT / "data" / "trajectories"
OUT_DIR = ROOT / "data" / "results" / "e8"


def coarse_weights(c: float, n: int) -> tuple[list[float], float]:
    """The n fine-noise weights and the normalisation that makes ξ_c standard normal."""
    w = [c ** (n - 1 - i) for i in range(n)]
    norm = math.sqrt((1.0 - c * c) / (1.0 - (c ** n) ** 2))
    return w, norm


def main() -> int:
    import h5py
    from ase import units
    from mace.calculators import mace_mp

    from frost.neighbours import VerletList
    from frost.teacher import frame_atoms
    from frost.teacher_fast import GraphedTeacher

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="silica1152_T300")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--start", type=int, default=9000)
    ap.add_argument("--fine-dt", type=float, default=1.0)
    ap.add_argument("--ratio", type=int, default=4, help="coarse step = ratio x fine step")
    ap.add_argument("--friction", type=float, default=0.05, help="per fs; the tuned baseline value")
    ap.add_argument("--fs", type=int, default=50000)
    ap.add_argument("--sample-every-fs", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--check", action="store_true", help="assert the coupling and exit")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    c = math.exp(-args.friction * args.fine_dt)
    w, norm = coarse_weights(c, args.ratio)
    residual = abs((1 - c * c) * sum(x * x for x in w) - (1 - (c ** args.ratio) ** 2))
    if residual > 1e-12:
        raise SystemExit(f"coupling is not exact: variance mismatch {residual:.3e}")
    if args.check:
        print(f"c={c:.8f} C={c ** args.ratio:.8f} weights={[round(x, 6) for x in w]} "
              f"norm={norm:.8f} variance residual={residual:.3e}")
        return 0

    dev = args.device
    temperature = float(args.run.split("_T")[1])
    with h5py.File(TRAJ / f"{args.run}.h5", "r") as fh:
        numbers = np.asarray(fh.attrs["numbers"], dtype=np.int64)
        cell = np.asarray(fh.attrs["cell"], dtype=np.float64)
        positions = np.asarray(fh["positions"][args.start], dtype=np.float64)
        velocities = np.asarray(fh["velocities"][args.start], dtype=np.float64)
    atoms = frame_atoms(numbers, positions, cell)
    n = len(atoms)
    masses = torch.as_tensor(atoms.get_masses(), dtype=torch.float64, device=dev).unsqueeze(1)
    x0 = torch.as_tensor(atoms.get_positions(), dtype=torch.float64, device=dev)
    v0 = torch.as_tensor(velocities, dtype=torch.float64, device=dev)
    sigma = torch.sqrt(units.kB * temperature / masses)

    calc = mace_mp(model="medium", device=dev, default_dtype="float32", enable_cueq=True)

    def make_arm(dt_fs: float):
        teacher = GraphedTeacher(calc, atoms, dev)
        verlet = VerletList(cell, teacher.r_max, 1.0, dev, wrap_positions=False, builder="alchemi")
        verlet.update(x0, force=True)
        teacher.set_edges(verlet.i, verlet.j, verlet.shifts, verlet.unit_shifts)

        def force(x):
            if verlet.update(x)[1]:
                teacher.set_edges(verlet.i, verlet.j, verlet.shifts, verlet.unit_shifts)
            e, f = teacher(x)
            return e, f.double()

        return {"x": x0.clone(), "v": v0.clone(), "force": force, "h": dt_fs * units.fs,
                "energies": [], "frames": []}

    fine, coarse = make_arm(args.fine_dt), make_arm(args.fine_dt * args.ratio)
    fine["f"] = fine["force"](fine["x"])[1]
    coarse["f"] = coarse["force"](coarse["x"])[1]
    c_coarse = c ** args.ratio

    def half_kick(arm, f):
        arm["v"] = arm["v"] + 0.5 * arm["h"] * f / masses

    def baoab(arm, noise, damp):
        """B-A-O-A-B with the O step's noise supplied, so both arms share one path."""
        half_kick(arm, arm["f"])
        arm["x"] = arm["x"] + 0.5 * arm["h"] * arm["v"]
        arm["v"] = damp * arm["v"] + math.sqrt(1.0 - damp * damp) * sigma * noise
        arm["x"] = arm["x"] + 0.5 * arm["h"] * arm["v"]
        e, f = arm["force"](arm["x"])
        arm["f"] = f
        half_kick(arm, f)
        return float(e) / n

    gen = torch.Generator(device=dev).manual_seed(args.seed)
    sample_every = max(1, round(args.sample_every_fs / args.fine_dt))
    n_coarse = args.fs // int(args.fine_dt * args.ratio)
    t0 = time.perf_counter()
    for block in range(n_coarse):
        draws = [torch.randn(x0.shape, generator=gen, device=dev, dtype=x0.dtype)
                 for _ in range(args.ratio)]
        for i, xi in enumerate(draws):
            e = baoab(fine, xi, c)
            if (block * args.ratio + i) % sample_every == 0:
                fine["energies"].append(e)
        combined = norm * sum(weight * xi for weight, xi in zip(w, draws, strict=True))
        coarse["energies"].append(baoab(coarse, combined, c_coarse))
        if block % 200 == 0:
            fine["frames"].append(fine["x"].to(torch.float32).cpu())
            coarse["frames"].append(coarse["x"].to(torch.float32).cpu())
    wall = time.perf_counter() - t0

    # the paired difference as a function of how far the two arms have run together
    fine_at_coarse = np.asarray(fine["energies"])[::max(1, args.ratio // sample_every)]
    length = min(len(fine_at_coarse), len(coarse["energies"]))
    diff = np.asarray(coarse["energies"][:length]) - fine_at_coarse[:length]
    growth = []
    for frac in (0.1, 0.25, 0.5, 1.0):
        m = int(frac * length)
        if m < 20:
            continue
        block_means = np.array([b.mean() for b in np.array_split(diff[:m], 10)])
        growth.append({"fraction": frac, "fs": m * args.fine_dt * args.ratio,
                       "paired_difference_meV": 1e3 * float(diff[:m].mean()),
                       "paired_se_meV": 1e3 * float(block_means.std(ddof=1) / math.sqrt(10))})

    report = {"run": args.run, "n_atoms": n, "config": vars(args),
              "coupling": {"c": c, "C": c_coarse, "weights": w, "norm": norm,
                           "variance_residual": residual},
              "wall_seconds": wall,
              "fine_mean_meV_per_atom": 1e3 * float(np.mean(fine["energies"])),
              "coarse_mean_meV_per_atom": 1e3 * float(np.mean(coarse["energies"])),
              "paired_by_length": growth}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"coupled_{args.run}_{args.tag}.json"
    out.write_text(json.dumps(report, indent=2))
    last = growth[-1] if growth else {}
    print(f"{args.run} {args.tag}: coarse-minus-fine {last.get('paired_difference_meV', float('nan')):+.4f}"
          f" +/- {last.get('paired_se_meV', float('nan')):.4f} meV/atom over {last.get('fs', 0):.0f} fs"
          f" | {wall:.0f} s | -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
