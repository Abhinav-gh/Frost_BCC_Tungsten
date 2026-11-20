"""In-situ student training from the teacher calls a multirate schedule makes anyway.

Every teacher call yields a labelled frame: positions, teacher energy, teacher
forces. Consecutive calls also give the change of the teacher force along the
actual motion, F_T(x_{n+1}) - F_T(x_n) ~ -H_T (x_{n+1} - x_n), a directional
Hessian at no extra cost. The loss trains the student on three things (note
Eq. 2):

* forces;
* those force changes, which set the refresh interval (Proposition 2);
* energy changes, which set the nested-verification acceptance (section 5.3).

All terms are in meV. The energy term is the squared total-energy error divided
by the atom count, so it grows with system size the way the acceptance does.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass
class Frame:
    """One teacher call, with the student's edges for those positions."""

    step: int
    x: torch.Tensor
    forces: torch.Tensor
    energy: float
    edge_i: torch.Tensor
    edge_j: torch.Tensor
    shifts: torch.Tensor


class LabelBuffer:
    """Teacher calls in arrival order; the oldest is dropped past capacity."""

    def __init__(self, capacity: int = 4096):
        self.capacity = capacity
        self.frames: list[Frame] = []

    def __len__(self) -> int:
        return len(self.frames)

    def add(self, frame: Frame) -> None:
        self.frames.append(frame)
        if len(self.frames) > self.capacity:
            del self.frames[0]

    def pairs(self, max_gap: int) -> list[tuple[int, int]]:
        """Consecutive calls close enough in time for their force change to be a label."""
        f = self.frames
        return [(a, a + 1) for a in range(len(f) - 1) if 0 < f[a + 1].step - f[a].step <= max_gap]

    def ring_pairs(self, stride: int) -> list[tuple[int, int]]:
        """Calls exactly a whole number of ``stride`` steps apart: the same moment on different path-integral beads.

        A bead dump (``pimd_runtime.py --dump-beads``) writes each bead's path as its own block of ``stride`` frames, so
        frames ``stride`` apart are ring partners at one time. Their force difference is the teacher's force change across
        the ring polymer, F_T(q_b) - F_T(q_b'), the structure a centroid-contracted residual cannot carry (2026-11-17: the
        quantum kinetic energy's shift fell from -0.47% at P' 1 to +0.04% at P' 16, so the student must carry it)."""
        index = {f.step: a for a, f in enumerate(self.frames)}
        out = []
        for a, f in enumerate(self.frames):
            m = 1
            while (b := index.get(f.step + m * stride)) is not None:
                out.append((a, b))
                m += 1
        return out

    def sample_pairs(self, count: int, max_gap: int,
                     generator: torch.Generator) -> list[tuple[int, int]]:
        """Random pairs of calls up to ``max_gap`` steps apart, not only consecutive ones.

        The first call is drawn uniformly among calls with a later call in range,
        and the second uniformly among those later calls. A stream of calls one
        step apart then still teaches the force change over a whole outer step,
        which is what the staleness certificate measures.
        """
        steps = [f.step for f in self.frames]
        ends = [bisect.bisect_right(steps, s + max_gap) for s in steps]
        anchors = [a for a, end in enumerate(ends) if end > a + 1]
        if not anchors:
            return []
        out = []
        for _ in range(count):
            a = anchors[int(torch.randint(len(anchors), (1,), generator=generator))]
            b = int(torch.randint(a + 1, ends[a], (1,), generator=generator))
            out.append((a, b))
        return out


def collate(frames: Sequence[Frame], types: torch.Tensor):
    """Frames of one system as a single disconnected graph."""
    n = frames[0].x.shape[0]
    x = torch.cat([f.x for f in frames])
    t = types.repeat(len(frames))
    ei = torch.cat([f.edge_i + k * n for k, f in enumerate(frames)])
    ej = torch.cat([f.edge_j + k * n for k, f in enumerate(frames)])
    sh = torch.cat([f.shifts for f in frames])
    batch = torch.arange(len(frames), device=x.device).repeat_interleave(n)
    return x, t, ei, ej, sh, batch


class InSituTrainer:
    """Adam on the three-term loss.

    With ``spread`` the force-change and energy-change labels come from any two
    calls up to ``max_gap`` steps apart (``LabelBuffer.sample_pairs``), not only
    from consecutive calls.
    """

    def __init__(self, student, types: torch.Tensor, lr: float = 2e-3, w_force: float = 1.0,
                 w_dforce: float = 1.0, w_denergy: float = 10.0, max_gap: int = 64, seed: int = 0,
                 spread: bool = False, ring_stride: int = 0, ring_batch: int = 0, species_balance: bool = False):
        self.student = student
        # species balance (2026-11-17): every element's mean force error counts equally. The solvated peptide is 22 of
        # 1,384 atoms, so a per-atom mean let water set the fit, and the six carbons' quantum kinetic energy stayed -1.3%
        # off after ring-pair training while hydrogen's halved. Weights average to one over the atoms.
        self.atom_w = None
        if species_balance:
            t = types.long()
            kinds, counts = torch.unique(t, return_counts=True)
            per = torch.zeros(int(t.max()) + 1, dtype=torch.float64, device=t.device)
            per[kinds] = t.numel() / (kinds.numel() * counts.double())
            self.atom_w = per[t]
        # ring pairs (bead dumps only): ``ring_batch`` pairs of ring partners per update, ``ring_stride`` frames apart,
        # join the time pairs in the change terms, so the student learns the force change across the ring as well
        self.ring_stride, self.ring_batch = ring_stride, ring_batch
        self._ring = None
        self.types = types
        self.weights = (w_force, w_dforce, w_denergy)
        self.max_gap = max_gap
        self.spread = spread
        self.opt = torch.optim.Adam(student.parameters(), lr=lr)
        self.gen = torch.Generator().manual_seed(seed)
        self.updates = 0

    def set_lr(self, lr: float) -> None:
        for group in self.opt.param_groups:
            group["lr"] = lr

    def step(self, buffer: LabelBuffer, batch_pairs: int = 4, report: bool = False) -> dict | None:
        """One update on random pairs of consecutive teacher calls.

        Returns None when the buffer has no pairs yet, an empty dict otherwise,
        and loss terms when ``report`` is set (which synchronises the GPU).
        """
        if self.spread:
            chosen = buffer.sample_pairs(batch_pairs, self.max_gap, self.gen)
        else:
            pairs = buffer.pairs(self.max_gap)
            picks = (torch.randint(len(pairs), (min(batch_pairs, len(pairs)),),
                                   generator=self.gen).tolist() if pairs else [])
            chosen = [pairs[p] for p in picks]
        if self.ring_stride and self.ring_batch:
            if self._ring is None or self._ring[0] != len(buffer):
                self._ring = (len(buffer), buffer.ring_pairs(self.ring_stride))
            ring = self._ring[1]
            if ring:
                picks = torch.randint(len(ring), (self.ring_batch,), generator=self.gen).tolist()
                chosen = chosen + [ring[p] for p in picks]
        if not chosen:
            return None
        first = [buffer.frames[a] for a, _ in chosen]
        second = [buffer.frames[b] for _, b in chosen]
        m = len(first)
        x, t, ei, ej, sh, batch = collate(first + second, self.types)
        n = first[0].x.shape[0]
        if getattr(self.student, "returns_forces", False):  # force-only student: forces are its output, no energy
            f_s = self.student(x, t, ei, ej, sh, batch, 2 * m).view(2 * m, n, 3)
            energy = f_s.new_zeros(2 * m)
        else:
            x = x.detach().requires_grad_(True)
            energy = self.student(x, t, ei, ej, sh, batch, 2 * m)
            (grad,) = torch.autograd.grad(energy.sum(), x, create_graph=True)
            f_s = -grad.view(2 * m, n, 3)
        f_t = torch.stack([f.forces for f in first + second]).to(f_s.dtype)
        e_t = torch.tensor([f.energy for f in first + second], dtype=energy.dtype,
                           device=energy.device)

        mev = 1e3
        w_atom = 1.0 if self.atom_w is None else self.atom_w.to(f_s.dtype).to(f_s.device)
        l_force = (((f_s - f_t) * mev).pow(2).sum(-1) * w_atom).mean()
        l_dforce = ((((f_s[m:] - f_s[:m]) - (f_t[m:] - f_t[:m])) * mev).pow(2).sum(-1) * w_atom).mean()
        l_denergy = (((energy[m:] - energy[:m]) - (e_t[m:] - e_t[:m])) * mev).pow(2).mean() / n
        w_f, w_d, w_e = self.weights
        if getattr(self.student, "returns_forces", False):
            w_e = 0.0  # no energy to fit
        loss = w_f * l_force + w_d * l_dforce + w_e * l_denergy

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self.updates += 1
        if not report:
            return {}
        return {
            "loss": float(loss.detach()),
            "force_rmse_meV_per_A": math.sqrt(float(l_force.detach())),
            "dforce_rmse_meV_per_A": math.sqrt(float(l_dforce.detach())),
            "denergy_rms_meV": math.sqrt(float(l_denergy.detach()) * n),
        }
