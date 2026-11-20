"""A pinned solid-liquid slab at temperature T, written as a trajectory a Frost run starts from (further_expansions_beyond.md
7.2: melting points by interface pinning, Pedersen 2013).

    python experiments/e17_melting/prep_slab.py --element Cu --temp 1300 --cells 6 6 20 --label cuslab

Steps, all with the teacher (MACE-MP-0 medium, graphed, its virial from the edge forces):
1. The crystal's lattice constant at T and P: a 4 x 4 x 4 bulk cell under isotropic stochastic cell rescaling and Langevin
   (4 ps settling, 4 ps averaged), and the crystal's Q at T (``frost.pinning``; the Debye-Waller factor lowers it below 1).
2. The liquid's volume per atom at T: the same cell melted at 2.2 T (3 ps), then at T under the barostat (3 + 4 ps).
3. The slab: NX x NY x NZ conventional cells at a(T), z normal to the interface; the upper half expanded along z by the
   liquid-to-solid volume ratio, then melted at 2.2 T (3 ps) and brought to T (3 ps) with the lower half frozen
   (positions held, velocities zero), so the crystal below keeps its lattice.
4. Release: the crystal gets Maxwell-Boltzmann velocities; NP_zAT (the lateral cell fixed at the crystal's lattice, L_z
   driven by P_zz) with the pinning bias at a = (Q_s + Q_l) / 2, 10 ps.
5. The trajectory: 10 ps more at the final cell under NVT with the bias, every step written (positions, velocities,
   energies, neighbour lists) to data/trajectories/<label>_T<T>.h5, so ``replay.py`` trains a student on it and
   ``md_runtime.py --start`` starts from its last frame. Parameters to data/results/e17/<label>_T<T>_prep.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TRAJ = ROOT / "data" / "trajectories"
OUT = ROOT / "data" / "results" / "e17"


class Sim:
    """The teacher on a cell that can move: forces, energy, the virial tensor, and set_cell for barostat moves."""

    def __init__(self, atoms, teacher_name: str, device: str = "cuda"):
        from frost.neighbours import VerletList
        from frost.teacher_fast import GraphedTeacher
        from frost.teachers import load_calc

        self.calc = load_calc(teacher_name, device, "float32", enable_cueq=True)
        self.teacher = GraphedTeacher(self.calc, atoms, device, edge_forces=True, virial=True)
        self.cell = np.array(atoms.cell.array, dtype=np.float64)
        self.list = VerletList(self.cell, self.teacher.r_max, 1.0, device, wrap_positions=False, builder="alchemi")
        self.seen = -1
        self.pin = None

    def force(self, x):
        self.list.update(x)
        if self.list.rebuilds != self.seen:
            self.seen = self.list.rebuilds
            self.teacher.set_edges(self.list.i, self.list.j, self.list.shifts, self.list.unit_shifts)
        e, f = self.teacher(x)
        f = f.double()
        if self.pin is not None:
            e_b, f_b, _ = self.pin(x)
            e, f = e + e_b, f + f_b
        return e, f

    def set_cell(self, cell):
        self.cell = np.array(cell, dtype=np.float64)
        self.list.set_cell(self.cell)
        self.teacher.set_edges(self.list.i, self.list.j, self.list.shifts, self.list.unit_shifts)

    def virial_tensor(self) -> np.ndarray:
        return self.teacher.static_virial_tensor.detach().double().cpu().numpy()


def run(sim, state, integ, steps: int, baro=None, every: int = 10, frozen=None, x_frozen=None, on_step=None):
    """Langevin steps with optional barostat moves (every ``every`` steps) and a frozen set (held in place, v = 0)."""
    e, f = sim.force(state.x)
    f = f.clone()
    vols = []
    for i in range(steps):
        e, f = integ.inner_step(lambda x: (*sim.force(x), x), f)
        f = f.clone()
        if frozen is not None:
            state.x[frozen] = x_frozen
            state.v[frozen] = 0.0
            f[frozen] = 0.0
        if baro is not None and (i + 1) % every == 0:
            vol = float(abs(np.linalg.det(sim.cell)))
            k_ax = (0.5 * (state.masses * state.v ** 2).sum(0)).double().cpu().numpy()
            mu = baro.scale_vector(vol, baro.pressure_axes(k_ax, sim.virial_tensor(), vol), every * integ.dt)
            mu_t = torch.as_tensor(mu, dtype=state.x.dtype, device=state.x.device)
            state.x = state.x * mu_t
            state.v = state.v / mu_t
            sim.set_cell(sim.cell * mu[None, :])
            e, f = sim.force(state.x)
            f = f.clone()
            vols.append(float(abs(np.linalg.det(sim.cell))))
        if on_step is not None:
            on_step(i, state, e)
    return vols


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--element", default="Cu")
    ap.add_argument("--a0", type=float, default=3.61, help="lattice constant to start from (A)")
    ap.add_argument("--temp", type=float, required=True)
    ap.add_argument("--cells", type=int, nargs=3, default=[6, 6, 20])
    ap.add_argument("--teacher", default="mp-medium")
    ap.add_argument("--kappa", type=float, default=500.0, help="pinning spring, eV (sd of Q ~ sqrt(kT / kappa))")
    ap.add_argument("--friction", type=float, default=0.01, help="per fs, preparation only")
    ap.add_argument("--traj-steps", type=int, default=10000)
    ap.add_argument("--label", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps-scale", type=float, default=1.0, help="multiply every phase's length (smoke tests)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(4)

    from ase.build import bulk
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    from frost.integrators import Integrator, Langevin, MDState, StochasticCellRescaling
    from frost.pinning import Pinning, inplane_kvectors
    from frost.trajectory import TrajectoryWriter

    out = TRAJ / f"{args.label}_T{int(args.temp)}.h5"
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists")
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    T = args.temp

    def ns(steps: int) -> int:
        return max(20, int(round(steps * args.steps_scale)))

    def region_q(x, mask, kvec) -> float:
        """Q of a subset of atoms (|sum exp(i k.r)| / n over the subset, averaged over k): ~1 crystal, ~n^-1/2 liquid."""
        k = torch.as_tensor(kvec, dtype=torch.float64, device=x.device)
        ph = x[mask].double() @ k.T
        return float((torch.sqrt(torch.cos(ph).sum(0) ** 2 + torch.sin(ph).sum(0) ** 2) / int(mask.sum())).mean())

    def limit_for(n_atoms: int, floor: float) -> float:
        """A melted region's Q is ~sqrt(pi / (4 n)) (random phases); allow three times that, never under ``floor``."""
        return max(floor, 3.0 * math.sqrt(math.pi / (4.0 * max(n_atoms, 1))))

    def melt_region(sim, st, masses, temp, seed, mask_of, kvec, frozen=None, x_frozen=None, floor=0.15):
        """Langevin at temp in 1 ps chunks until the region's Q is under its limit (at most 10 chunks); ``mask_of(x)``
        gives the region from the current positions."""
        integ = Integrator(st, 1.0, Langevin(1.0, args.friction, temp, masses, seed))
        for chunk in range(10):
            run(sim, st, integ, ns(1000), frozen=frozen, x_frozen=x_frozen)
            mask = mask_of(st.x)
            q = region_q(st.x, mask, kvec)
            if q < limit_for(int(mask.sum()), floor) and chunk >= 2:  # >= 3 ps: a fluctuation can dip early
                return q
        raise RuntimeError(f"the region did not melt at {temp:.0f} K in 10 ps (Q {q:.3f})")

    t0 = time.perf_counter()

    def bulk_phase(melt: bool):
        at = bulk(args.element, "fcc", a=args.a0, cubic=True).repeat((4, 4, 4))
        MaxwellBoltzmannDistribution(at, temperature_K=T * (2.2 if melt else 1.0), rng=np.random.default_rng(args.seed))
        sim = Sim(at, args.teacher, dev)
        masses = torch.as_tensor(at.get_masses(), dtype=torch.float64, device=dev).unsqueeze(1)
        st = MDState(torch.as_tensor(at.get_positions(), dtype=torch.float64, device=dev),
                     torch.as_tensor(at.get_velocities(), dtype=torch.float64, device=dev), masses)
        if melt:  # melt at constant volume first, checked by the crystal's own order parameter
            everyone = torch.ones(len(at), dtype=torch.bool, device=dev)
            melt_region(sim, st, masses, 2.2 * T, args.seed, lambda x: everyone, inplane_kvectors(sim.cell, (4, 4)),
                        floor=0.2)
        baro = StochasticCellRescaling(1.0, 500.0, 1e-5, T, args.seed)
        integ = Integrator(st, 1.0, Langevin(1.0, args.friction, T, masses, args.seed + 1))
        run(sim, st, integ, ns(3000) if melt else ns(4000), baro=baro)
        vols = run(sim, st, integ, ns(4000), baro=baro)
        v_atom = float(np.mean(vols)) / len(at)
        # Q at the final cell (its k vectors are exact for it), averaged over 2 ps at that fixed cell
        pin = Pinning(inplane_kvectors(sim.cell, (4, 4)), 0.0, 0.0, len(at), dev)
        qs = []
        run(sim, st, integ, ns(2000), on_step=lambda i, s, e: qs.append(float(pin.q(s.x))) if i % 10 == 0 else None)
        return v_atom, float(np.mean(qs))

    v_s, q_s = bulk_phase(melt=False)
    a_t = (4.0 * v_s) ** (1.0 / 3.0)  # four atoms per conventional fcc cell
    v_l, q_l_bulk = bulk_phase(melt=True)
    print(f"bulk at {T:.0f} K: a {a_t:.4f} A (v_s {v_s:.3f}), v_l {v_l:.3f} A^3/atom (ratio {v_l / v_s:.4f}), "
          f"Q crystal {q_s:.4f}, Q liquid (256 atoms) {q_l_bulk:.4f}  [{time.perf_counter() - t0:.0f} s]", flush=True)

    nx, ny, nz = args.cells
    slab = bulk(args.element, "fcc", a=a_t, cubic=True).repeat((nx, ny, nz))
    n = len(slab)
    lz = float(slab.cell[2, 2])
    pos = slab.get_positions()
    upper = pos[:, 2] >= 0.5 * lz - 1e-6
    ratio = v_l / v_s
    pos[upper, 2] = 0.5 * lz + (pos[upper, 2] - 0.5 * lz) * ratio
    cell = np.array(slab.cell.array)
    cell[2, 2] = 0.5 * lz * (1.0 + ratio)
    slab.set_cell(cell)
    slab.set_positions(pos)
    MaxwellBoltzmannDistribution(slab, temperature_K=2.2 * T, rng=np.random.default_rng(args.seed + 2))
    sim = Sim(slab, args.teacher, dev)
    masses = torch.as_tensor(slab.get_masses(), dtype=torch.float64, device=dev).unsqueeze(1)
    st = MDState(torch.as_tensor(pos, dtype=torch.float64, device=dev),
                 torch.as_tensor(slab.get_velocities(), dtype=torch.float64, device=dev), masses)
    frozen = torch.as_tensor(~upper, device=dev)
    x_frozen = st.x[frozen].clone()
    st.v[frozen] = 0.0
    kvec = inplane_kvectors(sim.cell, (nx, ny))
    z_lo, z_hi = 0.5 * lz + 5.0, float(cell[2, 2]) - 5.0  # the liquid's interior: 5 A clear of both crystal faces, where a
    if z_hi <= z_lo:                                       # liquid layers on the crystal template and is ordered anyway
        raise SystemExit("the liquid region is too thin for an interior (more cells along z)")

    def interior(x):
        z = torch.remainder(x[:, 2], float(cell[2, 2]))
        return (z > z_lo) & (z < z_hi) & ~frozen

    q_melt = melt_region(sim, st, masses, 2.2 * T, args.seed + 3, interior, kvec, frozen=frozen, x_frozen=x_frozen)
    run(sim, st, Integrator(st, 1.0, Langevin(1.0, args.friction, T, masses, args.seed + 4)), ns(3000),
        frozen=frozen, x_frozen=x_frozen)
    inner = interior(st.x)
    q_inner = region_q(st.x, inner, kvec)
    print(f"liquid interior ({int(inner.sum())} atoms) melted: Q {q_melt:.3f} at {2.2 * T:.0f} K, {q_inner:.3f} after 3 ps "
          f"at {T:.0f} K (limit {limit_for(int(inner.sum()), 0.25):.3f}); crystal Q {region_q(st.x, frozen, kvec):.3f}",
          flush=True)
    if q_inner > limit_for(int(inner.sum()), 0.25):
        raise RuntimeError(f"the liquid recrystallised while cooling (interior Q {q_inner:.3f})")
    q_l = math.sqrt(math.pi / (4.0 * n))  # a liquid's |rho_k| / N: the mean of a 2D Gaussian modulus, sqrt(pi N / 4) / N
    a_pin = 0.5 * (q_s + q_l)
    probe = Pinning(kvec, 0.0, 0.0, n, dev)
    print(f"slab {n} atoms, L_z {cell[2, 2]:.2f} A: melted; Q {float(probe.q(st.x)):.4f}, pin at a {a_pin:.4f} "
          f"(Q_s {q_s:.4f}, Q_l {q_l:.4f}), kappa {args.kappa} eV  [{time.perf_counter() - t0:.0f} s]", flush=True)

    # release under NP_zAT with the bias
    rng = torch.Generator(device=dev).manual_seed(args.seed + 5)
    st.v[frozen] = (torch.randn(st.v[frozen].shape, generator=rng, device=dev, dtype=torch.float64)
                    * torch.sqrt(8.617333262e-5 * T / masses[frozen]))
    sim.pin = Pinning(kvec, args.kappa, a_pin, n, dev)
    baro = StochasticCellRescaling(1.0, 1000.0, 1e-5, T, args.seed, axes="z")
    integ = Integrator(st, 1.0, Langevin(1.0, args.friction, T, masses, args.seed + 6))
    qs = []
    run(sim, st, integ, ns(10000), baro=baro,
        on_step=lambda i, s, e: qs.append(float(sim.pin.q(s.x))) if i % 20 == 0 else None)
    print(f"NP_zAT 10 ps: L_z {sim.cell[2, 2]:.2f} A, Q {np.mean(qs[-100:]):.4f} (sd {np.std(qs[-100:]):.4f}) "
          f"[{time.perf_counter() - t0:.0f} s]", flush=True)

    # the trajectory at the final cell (NVT, pinned): what replay trains on and md_runtime starts from
    final_cell = sim.cell.copy()
    slab.set_cell(final_cell)
    attrs = {"cell": final_cell, "numbers": slab.get_atomic_numbers().astype(np.uint8), "pbc": np.array(slab.pbc),
             "temperature_K": T, "timestep_fs": 1.0, "cutoff": 6.0, "skin": 1.0, "seed": args.seed,
             "teacher": args.teacher, "generator": "prep_slab.py", "source": f"{args.element} slab {nx}x{ny}x{nz}",
             "pin_kappa": args.kappa, "pin_a": a_pin, "pin_cells": [nx, ny]}
    from ase import units

    with TrajectoryWriter(out, n_atoms=n, cutoff=6.0, skin=1.0, attrs=attrs) as writer:
        def write(i, s, e):
            x_np = s.x.detach().cpu().numpy()
            slab.set_positions(x_np)
            slab.wrap()
            temp = float((s.masses * s.v ** 2).sum()) / (3 * n * units.kB)
            writer.write(i, x_np, slab, float(e), temp, s.v.detach().cpu().numpy().astype(np.float32))

        qs2 = []
        run(sim, st, integ, ns(args.traj_steps),
            on_step=lambda i, s, e: (write(i, s, e), qs2.append(float(sim.pin.q(s.x))) if i % 20 == 0 else None))
    info = {"element": args.element, "temperature_K": T, "a_A": a_t, "v_solid_A3": v_s, "v_liquid_A3": v_l,
            "q_solid": q_s, "q_liquid": q_l, "q_liquid_bulk256": q_l_bulk, "kappa_eV": args.kappa, "a_pin": a_pin,
            "cells": [nx, ny, nz], "n_atoms": n, "cell": final_cell.tolist(), "q_traj_mean": float(np.mean(qs2)),
            "q_traj_sd": float(np.std(qs2)), "trajectory": str(out), "seconds": time.perf_counter() - t0}
    (OUT / f"{args.label}_T{int(T)}_prep.json").write_text(json.dumps(info, indent=1))
    print(f"done: {out}; trajectory <Q> {info['q_traj_mean']:.4f} (a {a_pin:.4f})  [{info['seconds']:.0f} s]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
