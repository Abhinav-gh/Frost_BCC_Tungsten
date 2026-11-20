"""Teacher trajectories on the production teacher path instead of ASE (same file format as run_trajectory.py).

``run_trajectory.py`` drives MACE through ASE's calculator and neighbour list: 147 ms/step on 512-atom MgO and
43 ms/step on 648-atom water, most of it outside the model. This runs the same protocol -- Langevin NVT at
1 fs, friction 0.01/fs, equilibration discarded, then logged production -- with the teacher every arm already
uses (``GraphedTeacher``: cuEquivariance kernels in a CUDA graph, GPU Verlet lists) and our BAOAB integrator,
and writes the same HDF5 layout through ``TrajectoryWriter`` (positions unwrapped, velocities in ASE units,
energy, temperature, the logged neighbour lists, and the ``teacher`` attribute), so every downstream script
reads it unchanged.

``--tile-of RUN --tile N`` starts from frame ``--start`` of an existing trajectory replicated N^3 times (a
larger cell of an equilibrated structure); the equilibration then decorrelates the copies.

Usage (guarded launcher; GPU)
  scripts/safe_run.sh traj_cu4000 .venvs/frost-mace/bin/python experiments/e1_redundancy/fast_trajectory.py \\
      --tile-of cu500_T300 --tile 2 --temp 300 --label cu4000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "trajectories"


def last_dump_frame(path: Path, type_map: list[str]):
    """The last frame of a LAMMPS custom dump (``id type xu yu zu ...``, orthogonal box) as wrapped ASE atoms, ordered
    by atom id: e.g. Kanda et al.'s TiO2 after their NPT equilibration, so a run starts at their equilibrated volume."""
    from ase import Atoms

    last = None
    with path.open() as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            fh.readline()
            fh.readline()
            n = int(fh.readline())
            head = fh.readline()
            bounds = [list(map(float, fh.readline().split())) for _ in range(3)]
            if "xy" in head and any(abs(b[2]) > 1e-9 for b in bounds):
                raise SystemExit("triclinic dump box: not supported")
            cols = fh.readline().split()[2:]
            last = (bounds, cols, [fh.readline() for _ in range(n)])
    bounds, cols, rows = last
    data = np.loadtxt(rows)
    data = data[np.argsort(data[:, cols.index("id")])]
    lo = np.array([b[0] for b in bounds])
    box = np.array([b[1] - b[0] for b in bounds])
    x = np.mod(data[:, [cols.index("xu"), cols.index("yu"), cols.index("zu")]] - lo, box)
    symbols = [type_map[int(t) - 1] for t in data[:, cols.index("type")]]
    return Atoms(symbols=symbols, positions=x, cell=np.diag(box), pbc=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system", default="", help="an E1_SYSTEMS builder")
    ap.add_argument("--tile-of", default="", help="start from this run's frame, replicated")
    ap.add_argument("--tile", type=int, default=1)
    ap.add_argument("--start", type=int, default=9000)
    ap.add_argument("--temp", type=float, required=True)
    ap.add_argument("--teacher", default="mp-medium")
    ap.add_argument("--teacher-ext", default="",
                    help="a teacher served from its own environment (frost.external_teacher spec, e.g. kanda:<.pb>)")
    ap.add_argument("--from-dump", default="",
                    help="start from the last frame of a LAMMPS custom dump (orthogonal box, xu yu zu)")
    ap.add_argument("--type-map", nargs="*", default=["Ti", "O"], help="element of each LAMMPS atom type, in order")
    ap.add_argument("--equil", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--friction", type=float, default=0.01, help="per fs")
    ap.add_argument("--dt", type=float, default=1.0, help="fs (reactive hydrogen chemistry at ~3000 K wants ~0.25)")
    ap.add_argument("--cutoff", type=float, default=6.0, help="logged neighbour lists")
    ap.add_argument("--skin", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", required=True, help="run name: data/trajectories/<label>_T<temp>.h5")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from ase import units
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    from frost.integrators import Integrator, Langevin, MDState
    from frost.neighbours import VerletList
    from frost.systems import E1_SYSTEMS
    from frost.teacher import frame_atoms
    from frost.teacher_fast import GraphedTeacher
    from frost.teachers import load_calc
    from frost.trajectory import TrajectoryWriter

    out = OUT / f"{args.label}_T{int(args.temp)}.h5"
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists")
    dev = "cuda"
    if args.tile_of:
        with h5py.File(OUT / f"{args.tile_of}.h5", "r") as fh:
            numbers = np.asarray(fh.attrs["numbers"])
            cell = np.asarray(fh.attrs["cell"], dtype=np.float64)
            pos = np.asarray(fh["positions"][args.start], dtype=np.float64)
        offs = np.array([[a, b, c] for a in range(args.tile) for b in range(args.tile)
                         for c in range(args.tile)], dtype=np.float64) @ cell
        atoms = frame_atoms(np.concatenate([numbers] * len(offs)), np.concatenate([pos + o for o in offs]),
                            cell * args.tile)
    elif args.from_dump:
        atoms = last_dump_frame(Path(args.from_dump), args.type_map)
    else:
        atoms = E1_SYSTEMS[args.system]()
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temp, rng=np.random.default_rng(args.seed))
    n = len(atoms)
    cell = np.array(atoms.cell)
    masses = torch.as_tensor(atoms.get_masses(), dtype=torch.float64, device=dev).unsqueeze(1)
    x = torch.as_tensor(atoms.get_positions(), dtype=torch.float64, device=dev)
    v = torch.as_tensor(atoms.get_velocities(), dtype=torch.float64, device=dev)

    if args.teacher_ext:
        # a teacher in its own environment (frost.external_teacher): it keeps its own neighbour list
        from frost.external_teacher import ExternalTeacher

        ext = ExternalTeacher(args.teacher_ext, atoms.get_atomic_numbers(), cell, dev)

        def force(xx):
            e, f, _ = ext(xx)
            return float(e), f.double(), xx
    else:
        calc = load_calc(args.teacher, dev, "float32", enable_cueq=True)
        teacher = GraphedTeacher(calc, atoms, dev)
        t_list = VerletList(cell, teacher.r_max, args.skin, dev, wrap_positions=False, builder="alchemi")
        t_list.update(x, force=True)
        teacher.set_edges(t_list.i, t_list.j, t_list.shifts, t_list.unit_shifts)
        seen = {"rebuilds": t_list.rebuilds}

        def force(xx):
            t_list.update(xx)
            if t_list.rebuilds != seen["rebuilds"]:
                seen["rebuilds"] = t_list.rebuilds
                teacher.set_edges(t_list.i, t_list.j, t_list.shifts, t_list.unit_shifts)
            e, f = teacher(xx)
            return e, f.double(), xx

    state = MDState(x, v, masses)
    integ = Integrator(state, args.dt, Langevin(args.dt, args.friction, args.temp, masses, args.seed))
    energy, f, state.x = force(state.x)
    f = f.clone()

    def temperature() -> float:
        return float((masses * state.v ** 2).sum()) / (3 * n * units.kB)

    print(f"{args.label}: {n} atoms, teacher {args.teacher_ext or args.teacher}, {args.equil} + {args.steps} steps at {args.dt:g} fs, "
          f"Langevin {args.friction}/fs", flush=True)
    t0 = time.perf_counter()
    for s in range(1, args.equil + 1):
        energy, f = integ.inner_step(force, f)
        f = f.clone()
        if s % 500 == 0:
            el = time.perf_counter() - t0
            t_now = temperature()
            if not np.isfinite(t_now):
                raise RuntimeError("non-finite state during equilibration; check the cell (tests/unit/test_systems.py)")
            print(f"  equilibrating {s}/{args.equil}  {el:.0f}s  {el / s * 1e3:.1f} ms/step  T={t_now:.0f}K", flush=True)

    # one byte per atomic number: HDF5 caps an attribute at 64 KB, which int64 numbers pass at 8,192 atoms (every
    # reader casts to int64)
    attrs = {"cell": cell, "numbers": atoms.get_atomic_numbers().astype(np.uint8), "pbc": np.array(atoms.pbc),
             "temperature_K": args.temp, "timestep_fs": args.dt, "cutoff": args.cutoff, "skin": args.skin,
             "seed": args.seed, "teacher": args.teacher_ext or args.teacher, "generator": "fast_trajectory.py",
             "source": args.from_dump or args.system or args.tile_of}
    t0 = time.perf_counter()
    n_rebuild = 0
    with TrajectoryWriter(out, n_atoms=n, cutoff=args.cutoff, skin=args.skin, attrs=attrs) as writer:
        for s in range(args.steps):
            energy, f = integ.inner_step(force, f)
            f = f.clone()
            x_np = state.x.detach().cpu().numpy()  # the integrator never wraps: these are unwrapped already
            atoms.set_positions(x_np)
            atoms.wrap()  # the logged neighbour list is built on wrapped positions
            n_rebuild += writer.write(s, x_np, atoms, float(energy), temperature(),
                                      state.v.detach().cpu().numpy().astype(np.float32))
            if s and s % 1000 == 0:
                el = time.perf_counter() - t0
                print(f"  step {s}/{args.steps}  {el:.0f}s  {el / s * 1e3:.1f} ms/step  rebuilds={n_rebuild}  "
                      f"T={temperature():.0f}K", flush=True)
    print(f"done -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
