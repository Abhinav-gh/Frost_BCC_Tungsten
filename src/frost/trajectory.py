"""Trajectory logging for E1, the Gate 1 measurement.

What gets logged, and why it is not what the implementation guide suggested.

The guide proposes logging edge vectors for a random 10% of edges, on the
grounds that the full set is too large.  It is: for Si512 at 23,552 edges,
10,000 steps of float32 edge vectors is 2.8 GB per run, and there are ten runs.

But edge vectors are a *derived* quantity.  Logging **unwrapped positions**
plus the neighbour list at its rebuild points is sufficient to reconstruct
every edge vector exactly, and for the same system it is 60 MB rather than
2.8 GB -- a factor of 47, with no subsampling and therefore no sampling noise
in the dirty fraction.

Unwrapped is the operative word.  The predicate thresholds on a difference
across time, so a wrapped coordinate injects a spurious box-length jump the
moment an atom crosses a face.  `frost.edgekey.ImageTracker` is what keeps the
logged positions continuous; see `docs/design/plan.md` section 6.1.

Schema (HDF5):

    /positions      (n_steps, n_atoms, 3)  float64, UNWRAPPED
    /velocities     (n_steps, n_atoms, 3)  float32, optional
    /energy         (n_steps,)             float64, potential energy
    /temperature    (n_steps,)             float32
    /nl/step        (n_rebuilds,)          int64,  step at which the list was built
    /nl/offset      (n_rebuilds + 1,)      int64,  index into i/j/S per rebuild
    /nl/i,/nl/j     (total_edges,)         int32
    /nl/S           (total_edges, 3)       int8
    attrs: system, model, temperature_K, timestep_fs, cutoff, cell, numbers, ...
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from frost.edgekey import ImageTracker

if TYPE_CHECKING:
    from ase import Atoms


def neighbour_list(atoms: Atoms, cutoff: float):
    """(i, j, S) for the current geometry, via vesin when available.

    vesin is roughly 47x faster than ASE's implementation on a 512-atom cell,
    which matters because E1 rebuilds this thousands of times.
    """
    try:
        from vesin import NeighborList

        nl = NeighborList(cutoff=cutoff, full_list=True)
        i, j, S = nl.compute(
            points=np.ascontiguousarray(atoms.get_positions()),
            box=np.ascontiguousarray(np.array(atoms.cell)),
            periodic=bool(atoms.pbc.all()),
            quantities="ijS",
        )
        return (
            np.asarray(i, np.int32),
            np.asarray(j, np.int32),
            np.asarray(S, np.int8),
        )
    except Exception:
        from ase.neighborlist import neighbor_list as ase_nl

        i, j, S = ase_nl("ijS", atoms, cutoff)
        return (
            np.asarray(i, np.int32),
            np.asarray(j, np.int32),
            np.asarray(S, np.int8),
        )


@dataclass
class TrajectoryWriter:
    """Append-mode HDF5 writer with a Verlet-skin neighbour-list policy."""

    path: Path
    n_atoms: int
    cutoff: float
    skin: float = 1.0
    attrs: dict = field(default_factory=dict)

    def __post_init__(self):
        import h5py

        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = h5py.File(self.path, "w")
        c = self._f.create_dataset
        kw = {"compression": "gzip", "compression_opts": 1, "shuffle": True}
        self._pos = c(
            "positions",
            (0, self.n_atoms, 3),
            maxshape=(None, self.n_atoms, 3),
            dtype="f8",
            chunks=(64, self.n_atoms, 3),
            **kw,
        )
        self._vel = c(
            "velocities",
            (0, self.n_atoms, 3),
            maxshape=(None, self.n_atoms, 3),
            dtype="f4",
            chunks=(64, self.n_atoms, 3),
            **kw,
        )
        self._ene = c("energy", (0,), maxshape=(None,), dtype="f8", chunks=(1024,))
        self._tmp = c("temperature", (0,), maxshape=(None,), dtype="f4", chunks=(1024,))
        g = self._f.create_group("nl")
        self._nl_step = g.create_dataset("step", (0,), maxshape=(None,), dtype="i8")
        self._nl_off = g.create_dataset("offset", (1,), maxshape=(None,), dtype="i8")
        self._nl_i = g.create_dataset("i", (0,), maxshape=(None,), dtype="i4", **kw)
        self._nl_j = g.create_dataset("j", (0,), maxshape=(None,), dtype="i4", **kw)
        self._nl_S = g.create_dataset("S", (0, 3), maxshape=(None, 3), dtype="i1", **kw)
        self._nl_off[0] = 0
        for k, v in self.attrs.items():
            self._f.attrs[k] = v
        self._n = 0
        self._ref_pos = None  # positions at the last neighbour-list build

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _append(ds, value):
        n = ds.shape[0]
        ds.resize(n + 1, axis=0)
        ds[n] = value

    def _needs_rebuild(self, unwrapped: np.ndarray) -> bool:
        """Standard Verlet criterion: rebuild once any atom has moved skin/2."""
        if self._ref_pos is None:
            return True
        d = np.linalg.norm(unwrapped - self._ref_pos, axis=1).max()
        return bool(d > 0.5 * self.skin)

    # -- public ------------------------------------------------------------
    def write(
        self,
        step: int,
        unwrapped: np.ndarray,
        atoms: Atoms,
        energy: float,
        temperature: float,
        velocities=None,
    ) -> bool:
        rebuilt = False
        if self._needs_rebuild(unwrapped):
            i, j, S = neighbour_list(atoms, self.cutoff + self.skin)
            n0, m = self._nl_i.shape[0], len(i)
            for ds, val in ((self._nl_i, i), (self._nl_j, j)):
                ds.resize(n0 + m, axis=0)
                ds[n0:] = val
            self._nl_S.resize(n0 + m, axis=0)
            self._nl_S[n0:] = S
            self._append(self._nl_step, step)
            k = self._nl_off.shape[0]
            self._nl_off.resize(k + 1, axis=0)
            self._nl_off[k] = n0 + m
            self._ref_pos = unwrapped.copy()
            rebuilt = True

        n = self._pos.shape[0]
        self._pos.resize(n + 1, axis=0)
        self._pos[n] = unwrapped
        if velocities is not None:
            self._vel.resize(n + 1, axis=0)
            self._vel[n] = velocities
        self._append(self._ene, energy)
        self._append(self._tmp, temperature)
        self._n += 1
        return rebuilt

    def close(self):
        self._f.attrs["n_steps"] = self._n
        self._f.attrs["n_rebuilds"] = int(self._nl_step.shape[0])
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class TrajectoryReader:
    """Read a logged trajectory and reconstruct edge geometry on demand."""

    def __init__(self, path: str | Path):
        import h5py

        self._f = h5py.File(path, "r")
        self.attrs = dict(self._f.attrs)
        self.positions = self._f["positions"]
        self.energy = self._f["energy"]
        self.temperature = self._f["temperature"]
        self.cell = np.asarray(self.attrs["cell"], dtype=np.float64)
        self.numbers = np.asarray(self.attrs["numbers"], dtype=np.int64)
        self._nl_step = np.asarray(self._f["nl/step"])
        self._nl_off = np.asarray(self._f["nl/offset"])
        self.n_steps = int(self.attrs["n_steps"])
        self.n_atoms = self.positions.shape[1]
        self._nl_cache: dict[int, tuple] = {}

    def generation_at(self, step: int) -> int:
        """Index of the neighbour list in force at `step`.

        The edge set is constant within a generation, which is what lets the
        cache replay work on plain arrays instead of a dictionary keyed by edge.
        """
        return max(int(np.searchsorted(self._nl_step, step, side="right") - 1), 0)

    def neighbour_list_at(self, step: int):
        """(i, j, S, generation) for the list in force at `step`.

        Cached per generation: without this the sweep re-reads the same slice
        out of HDF5 once per step per tolerance.
        """
        k = self.generation_at(step)
        hit = self._nl_cache.get(k)
        if hit is not None:
            return hit
        a, b = int(self._nl_off[k]), int(self._nl_off[k + 1])
        out = (
            np.asarray(self._f["nl/i"][a:b], np.int64),
            np.asarray(self._f["nl/j"][a:b], np.int64),
            np.asarray(self._f["nl/S"][a:b], np.int64),
            k,
        )
        # Only a handful of generations are ever live in one pass.
        if len(self._nl_cache) > 4:
            self._nl_cache.clear()
        self._nl_cache[k] = out
        return out

    def edge_vectors_from(self, positions: np.ndarray, step: int) -> np.ndarray:
        """r_ij for the list in force at `step`, given positions already read."""
        from frost.edgekey import edge_vectors

        i, j, S, _ = self.neighbour_list_at(step)
        return edge_vectors(positions, i, j, S, self.cell)

    def edge_vectors_at(self, step: int):
        """r_ij for every edge in the list in force at `step`."""
        from frost.edgekey import edge_vectors

        i, j, S, gen = self.neighbour_list_at(step)
        pos = np.asarray(self.positions[step])
        return edge_vectors(pos, i, j, S, self.cell), i, j, S, gen

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def run_md(
    atoms,
    calc,
    *,
    steps: int,
    timestep_fs: float,
    temperature_K: float,
    out: Path | None,
    cutoff: float,
    skin: float = 1.0,
    equilibrate_steps: int = 0,
    friction_per_fs: float = 0.01,
    log_every: int = 1,
    seed: int = 0,
    progress_every: int = 500,
    extra_attrs: dict | None = None,
):
    """NVT Langevin with unwrapped-position logging.

    Equilibration runs first and is discarded, per the E1 protocol.
    """
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    atoms = atoms.copy()
    atoms.calc = calc
    MaxwellBoltzmannDistribution(
        atoms, temperature_K=temperature_K, rng=np.random.default_rng(seed)
    )

    dyn = Langevin(
        atoms,
        timestep_fs * units.fs,
        temperature_K=temperature_K,
        friction=friction_per_fs / units.fs,
        rng=np.random.default_rng(seed),
    )

    if equilibrate_steps:
        # Report progress during equilibration, not only after it. A run that
        # prints nothing for 20,000 steps is indistinguishable from a run that
        # has wedged on a degenerate geometry, and that ambiguity once cost two
        # silica runs nineteen hours before anyone looked at GPU utilisation.
        t0 = time.perf_counter()
        chunk = max(1, min(progress_every or equilibrate_steps, equilibrate_steps))
        done = 0
        while done < equilibrate_steps:
            n = min(chunk, equilibrate_steps - done)
            dyn.run(n)
            done += n
            el = time.perf_counter() - t0
            if not np.isfinite(atoms.get_potential_energy()):
                raise RuntimeError(
                    "energy is not finite during equilibration; the starting "
                    "geometry is almost certainly degenerate. Check the cell "
                    "with tests/unit/test_systems.py before rerunning."
                )
            print(
                f"  equilibrating {done}/{equilibrate_steps}  {el:.0f}s  "
                f"{el / done * 1e3:.1f} ms/step  T={atoms.get_temperature():.0f}K",
                flush=True,
            )
        print(f"  equilibrated {equilibrate_steps} steps in {el:.0f}s")

    if out is None:
        return atoms

    tracker = ImageTracker(atoms.get_positions(), np.array(atoms.cell))
    writer = TrajectoryWriter(
        out,
        n_atoms=len(atoms),
        cutoff=cutoff,
        skin=skin,
        attrs={
            "cell": np.array(atoms.cell),
            "numbers": atoms.get_atomic_numbers(),
            "pbc": np.array(atoms.pbc),
            "temperature_K": temperature_K,
            "timestep_fs": timestep_fs,
            "cutoff": cutoff,
            "skin": skin,
            "seed": seed,
            **(extra_attrs or {}),
        },
    )
    t0 = time.perf_counter()
    n_rebuild = 0
    with writer:
        for s in range(steps):
            dyn.run(1)
            if s % log_every:
                continue
            unwrapped = tracker.update(atoms.get_positions())
            n_rebuild += writer.write(
                s,
                unwrapped,
                atoms,
                float(atoms.get_potential_energy()),
                float(atoms.get_temperature()),
                atoms.get_velocities().astype(np.float32),
            )
            if progress_every and s and s % progress_every == 0:
                el = time.perf_counter() - t0
                print(
                    f"  step {s}/{steps}  {el:.0f}s  "
                    f"{el / s * 1e3:.1f} ms/step  rebuilds={n_rebuild}  "
                    f"T={atoms.get_temperature():.0f}K",
                    flush=True,
                )
    el = time.perf_counter() - t0
    print(
        f"  done {steps} steps in {el:.0f}s ({el / steps * 1e3:.1f} ms/step), "
        f"{n_rebuild} rebuilds -> {out}"
    )
    return atoms
