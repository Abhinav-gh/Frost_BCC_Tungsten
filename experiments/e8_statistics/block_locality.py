"""E8: how far does a block move's energy change reach?

Block moves are the one structurally different route left for the statistics
contract (plan section 1.14): whole-cell proposals carry an extensive energy
residual, so acceptance falls with N, and moving one block at a time is the
standard fix. It only pays if the block's energy change can be evaluated locally,
and whether it can is a property of the teacher, not of the sampler.

That property has been asserted twice in this plan from arithmetic — MACE-MP
medium has two interaction layers and r_max 6 A, so the receptive field is 12 A,
so a block needs a 24 A halo and no cell we have is big enough. This measures it
instead. A block of atoms is displaced the way a draft would displace it, and the
teacher's *per-atom* energies are compared before and after across the whole cell:

    |de_i| against the distance of atom i from the block

The halo is the distance beyond which |de_i| falls into the noise of the
untouched region. Two teacher calls per block, every atom measured, no
single-atom probes.

What the answer decides: a block move needs a halo of that size for an exact
local energy difference, and two blocks are independent only if the cell holds
both with their halos. If the measured reach is well under 12 A, block moves are
possible on cells we already have.

Output: data/results/e8/block_locality_<run>[_<tag>].json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TRAJ = ROOT / "data" / "trajectories"
OUT_DIR = ROOT / "data" / "results" / "e8"


def minimum_image(delta: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """Displacements wrapped into the cell, for distances in a periodic box."""
    fractional = torch.linalg.solve(cell.T, delta.T).T
    return (fractional - fractional.round()) @ cell


def main() -> int:
    import h5py
    from mace.calculators import mace_mp

    from frost.neighbours import VerletList
    from frost.teacher import frame_atoms

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="silica1152_T300")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--start", type=int, default=9000)
    ap.add_argument("--block-radius", type=float, default=6.0, help="A; atoms within this move")
    ap.add_argument("--amplitude", type=float, default=0.1, help="A; rms displacement of the block")
    ap.add_argument("--blocks", type=int, default=8, help="independent block centres to average")
    ap.add_argument("--bin-width", type=float, default=1.0, help="A")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dev = args.device

    with h5py.File(TRAJ / f"{args.run}.h5", "r") as fh:
        numbers = np.asarray(fh.attrs["numbers"], dtype=np.int64)
        cell_np = np.asarray(fh.attrs["cell"], dtype=np.float64)
        positions = np.asarray(fh["positions"][args.start], dtype=np.float64)
    atoms = frame_atoms(numbers, positions, cell_np)
    n = len(atoms)
    cell = torch.as_tensor(cell_np, dtype=torch.float64, device=dev)
    x0 = torch.as_tensor(atoms.get_positions(), dtype=torch.float64, device=dev)

    calc = mace_mp(model="medium", device=dev, default_dtype="float32", enable_cueq=True)
    model = calc.models[0]
    r_max = float(model.r_max)
    from frost.calibrate import _batch_for

    template = _batch_for(calc, atoms, r_max, dev)[0].to_dict()
    lists = VerletList(cell_np, r_max, 0.0, dev, wrap_positions=False, builder="alchemi")

    def site_energies(x: torch.Tensor) -> torch.Tensor:
        """Per-atom energies on a list rebuilt for these positions."""
        lists.update(x, force=True)
        data = dict(template)
        data["positions"] = x.to(template["positions"].dtype)
        data["edge_index"] = torch.stack([lists.i, lists.j])
        data["shifts"] = lists.shifts.to(template["positions"].dtype)
        data["unit_shifts"] = lists.unit_shifts.to(template["positions"].dtype)
        with torch.no_grad():
            out = model(data, compute_force=False, training=False)
        return out["node_energy"].detach().double()

    gen = torch.Generator(device=dev).manual_seed(0)
    edges = float(min(torch.linalg.norm(cell, dim=1)))
    max_distance = 0.5 * edges  # beyond half the shortest edge, the image is nearer
    bins = np.arange(0.0, max_distance + args.bin_width, args.bin_width)
    profiles, tails, block_sizes, totals = [], [], [], []

    base = site_energies(x0)
    for _ in range(args.blocks):
        centre = x0[int(torch.randint(n, (), generator=gen, device=dev))]
        distance = minimum_image(x0 - centre, cell).norm(dim=1)
        in_block = distance <= args.block_radius
        block_sizes.append(int(in_block.sum()))

        x1 = x0.clone()
        shift = args.amplitude * torch.randn((int(in_block.sum()), 3), generator=gen,
                                             device=dev, dtype=x0.dtype)
        x1[in_block] += shift
        changed = (site_energies(x1) - base).abs() * 1e3  # meV

        # distance from the block's surface, so the profile starts where the block ends
        outside = (distance - args.block_radius).clamp(min=0.0).cpu().numpy()
        index = np.digitize(outside, bins) - 1
        values = changed.cpu().numpy()
        profile = [float(values[index == k].mean()) if (index == k).any() else float("nan")
                   for k in range(len(bins) - 1)]
        # what decides a halo is not the mean per atom but the total energy left
        # outside it, which is what truncating the halo would get wrong
        tail = [float(values[outside >= bins[k]].sum()) for k in range(len(bins) - 1)]
        profiles.append(profile)
        tails.append(tail)
        totals.append(float(changed.sum()))

    mean_profile = np.nanmean(np.array(profiles), axis=0)
    mean_tail = np.nanmean(np.array(tails), axis=0)
    kT_meV = 1e3 * 8.617333262e-5 * float(args.run.split("_T")[1])
    # the halo an exact-enough local test needs: the first radius whose truncated
    # tail is under a hundredth of kT, which is negligible beside the model
    # residual the acceptance test already carries (~0.8 kT over a draft)
    halo = next((float(bins[k]) for k, v in enumerate(mean_tail) if v <= 0.01 * kT_meV), float("nan"))
    # the noise floor: the outermost bins, which no receptive field can reach
    floor = float(np.nanmean(mean_profile[-3:]))
    reach = float("nan")
    for k, value in enumerate(mean_profile):
        if np.isfinite(value) and value <= max(2.0 * floor, floor + 1e-6):
            reach = float(bins[k])
            break

    report = {
        "run": args.run, "n_atoms": n, "r_max_A": r_max, "layers": 2,
        "cell_edges_A": [float(v) for v in torch.linalg.norm(cell, dim=1)],
        "config": vars(args),
        "block_atoms_mean": float(np.mean(block_sizes)),
        "bins_A": [float(v) for v in bins[:-1]],
        "mean_abs_site_energy_change_meV": [float(v) for v in mean_profile],
        "noise_floor_meV": floor,
        "reach_A": reach,
        "kT_meV": kT_meV,
        "tail_energy_beyond_r_meV": [float(v) for v in mean_tail],
        "tail_energy_beyond_r_kT": [float(v / kT_meV) for v in mean_tail],
        "halo_needed_A": halo,
        "cell_fits_two_independent_blocks": bool(
            np.isfinite(halo) and 2.0 * (args.block_radius + halo) <= float(min(torch.linalg.norm(cell, dim=1)))),
        "cell_edge_needed_A": float(2.0 * (args.block_radius + halo)) if np.isfinite(halo) else float("nan"),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out = OUT_DIR / f"block_locality_{args.run}{suffix}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"{args.run}: block r={args.block_radius} A ({report['block_atoms_mean']:.0f} atoms) | "
          f"halo {halo:.1f} A for a tail under 0.01 kT | tail beyond r, in kT: "
          + " ".join(f"{b:.0f}A:{v / kT_meV:.3f}" for b, v in zip(bins[:-1], mean_tail, strict=False))
          + f" | needs a {report['cell_edge_needed_A']:.0f} A edge, cell has "
            f"{min(report['cell_edges_A']):.1f}", flush=True)
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
