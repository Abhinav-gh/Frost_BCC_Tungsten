"""The expensive model: MACE driven directly on MACE batches, and timed.

Driving the model directly rather than through ASE matters for speed and for
honest timing: ASE's own neighbour list costs more than the model on these
cells (plan section 1.3).
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable

import numpy as np
import torch


def frame_atoms(numbers, positions, cell):
    """A periodic ASE Atoms for one stored frame, wrapped into the cell."""
    from ase import Atoms

    atoms = Atoms(
        numbers=numbers,
        positions=np.asarray(positions, dtype=np.float64),
        cell=cell,
        pbc=True,
    )
    atoms.wrap()
    return atoms


def mace_forces(calc, atoms, device: str):
    """Forces (eV/A, float64 numpy), energy (eV) and the MACE batch for one frame."""
    from frost.calibrate import _batch_for

    model = calc.models[0]
    batch = _batch_for(calc, atoms, float(model.r_max), device)[0]
    out = model(batch.to_dict(), compute_force=True, training=False)
    forces = out["forces"].detach().double().cpu().numpy()
    energy = float(out["energy"].detach().double().sum().cpu())
    return forces, energy, batch


def time_call_ms(fn: Callable[[], object], repeats: int = 10, warmup: int = 5) -> float:
    """Median wall time of a GPU call in milliseconds, synchronised with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return float(np.median(times))


def time_mace_ms(model, batch, force: bool = True, repeats: int = 10, warmup: int = 5) -> float:
    """A MACE force call (forward+backward), or an energy-only forward under no_grad."""
    data = batch.to_dict()
    ctx = contextlib.nullcontext() if force else torch.no_grad()
    with ctx:
        return time_call_ms(
            lambda: model(data, compute_force=force, training=False), repeats, warmup
        )


class MaceTeacher:
    """MACE on a shared Verlet list: positions in, energy and forces out.

    The input dictionary comes from one MACE batch of the starting frame, so
    every key the model reads is present. Only positions change between list
    rebuilds; edges and shifts change at rebuilds. Edges between r_max and
    r_max + skin are inert, because MACE multiplies every radial feature by an
    envelope that is exactly zero beyond r_max
    (mace.modules.radial.PolynomialCutoff). One list therefore serves the teacher
    and the student.
    """

    def __init__(self, calc, atoms, device: str = "cuda"):
        from frost.calibrate import _batch_for

        self.model = calc.models[0]
        self.r_max = float(self.model.r_max)
        batch = _batch_for(calc, atoms, self.r_max, device)[0]
        self.data = batch.to_dict()
        self.dtype = self.data["positions"].dtype

    def set_edges(self, i: torch.Tensor, j: torch.Tensor, shifts: torch.Tensor,
                  unit_shifts: torch.Tensor) -> None:
        self.data["edge_index"] = torch.stack((i, j))
        self.data["shifts"] = shifts.to(self.dtype)
        self.data["unit_shifts"] = unit_shifts.to(self.dtype)

    def __call__(self, x: torch.Tensor):
        self.data["positions"] = x.detach().to(self.dtype).clone()
        out = self.model(self.data, compute_force=True, training=False)
        return out["energy"].detach().sum(), out["forces"].detach()

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        self.data["positions"] = x.detach().to(self.dtype).clone()
        with torch.no_grad():
            out = self.model(self.data, compute_force=False, training=False)
        return out["energy"].detach().sum()
