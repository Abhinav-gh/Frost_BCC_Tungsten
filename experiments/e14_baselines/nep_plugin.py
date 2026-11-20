"""MACE-NEP's own distilled NEP as Frost's student (``validate.py --student-plugin``; Part D, D2).

The NEP is the one ``mace_nep.py train`` distilled from MACE-MP-0 medium (``nep_best.txt``), evaluated by
TorchNEP's own descriptor and network operators on Frost's GPU Verlet list: pair vectors
r_ij = x[j] - x[i] + shift (TorchNEP's convention, ``torchnep.neighbor``), masked to the radial and angular
cutoffs, energy from ``ops.compute_descriptors`` and ``ops.apply_ann``, forces from autograd on the pair
vectors summed by ``ops.accumulate_forces_virial`` -- the path of ``NEPCalculator.compute`` with the
neighbour search replaced by ours. TorchNEP is GPL-3.0: this file runs only in ``.venvs/frost-nep``, and
nothing under ``src/frost`` imports it.

Usage
  scripts/safe_run.sh nep_student .venvs/frost-nep/bin/python experiments/e6_dynamics/validate.py \\
      --run cu500_T300 --student data/baselines/mace_nep/cu500_T300/nep/nep_best.txt \\
      --student-plugin experiments/e14_baselines/nep_plugin.py --k 24 --fs 50000 --tag nep_k24_50ps
"""

from __future__ import annotations

import torch
from torchnep import ops
from torchnep.nep import NEPCalculator

SYMBOLS = ["X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
           "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"]


class NEPStudent:
    def __init__(self, path: str, device: str):
        self.calc = NEPCalculator(path, dtype=torch.float32, device=device)
        self.device = device
        self.rc_r, self.rc_a = self.calc.cutoff_args()
        # TorchNEP's fastest eager GPU path (ops.resolve_backend: "mulsum" for <= 32 types); "loop" is 2-30x slower
        self.backend = ops.resolve_backend("auto", num_types=self.calc.num_types, device_type="cuda")
        self.config = {"kind": "nep", "path": path, "cutoff": float(max(self.calc.rc_radial, self.calc.rc_angular)),
                       "type_names": list(self.calc.type_names)}

    def types(self, numbers: torch.Tensor) -> torch.Tensor:
        names = [SYMBOLS[int(z)] for z in numbers.tolist()]
        return torch.tensor([self.calc.type_names.index(s) for s in names], dtype=torch.long, device=self.device)

    def graphed(self, types: torch.Tensor, n_atoms: int, capacity: int):
        self.atom_types, self.n = types, n_atoms
        self.capacity, self.graph = 0, None
        return self

    def _allocate(self, capacity: int) -> None:
        dev = self.atom_types.device
        self.capacity = capacity
        self.x = torch.zeros(self.n, 3, dtype=torch.float32, device=dev)
        self.ib = torch.zeros(capacity, dtype=torch.long, device=dev)
        self.jb = torch.zeros(capacity, dtype=torch.long, device=dev)
        self.sb = torch.zeros(capacity, 3, dtype=torch.float32, device=dev)
        self.graph = None

    def set_edges(self, i: torch.Tensor, j: torch.Tensor, shifts: torch.Tensor) -> None:
        n = int(i.numel())
        if n > self.capacity:
            self._allocate(int(1.25 * n) + 1)
        with torch.no_grad():
            self.ib.zero_()
            self.jb.zero_()
            self.sb.zero_()
            self.sb[n:, 0] = 1.0e3  # padding: atom 0 to an image far away, moved onto the cutoff sphere below
            self.ib[:n], self.jb[:n], self.sb[:n] = i, j, shifts.to(torch.float32)
        if self.graph is None:
            self._capture()

    def _capture(self) -> None:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._call()
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(self.graph):
                self.static = self._call()
        except Exception as exc:  # an op TorchNEP runs on the host: stay eager
            self.graph = False
            self.capture_error = repr(exc)[:500]

    def __call__(self, x: torch.Tensor):
        with torch.no_grad():
            self.x.copy_(x.to(torch.float32))
        if self.graph:
            self.graph.replay()
            return self.static
        return self._call()

    def _call(self):
        c = self.calc
        rij = self.x.index_select(0, self.jb) - self.x.index_select(0, self.ib) + self.sb
        d = rij.norm(dim=-1, keepdim=True)
        # Static shapes, for a CUDA graph: every list edge enters both pair sets, and a pair at or beyond a
        # cutoff is moved onto that cutoff's sphere, where TorchNEP's basis 0.5 (T_k + 1) f_c and its radial
        # derivative both vanish (f_c(rc) = 0, f_c'(rc) = 0), so it adds nothing to the energy or the forces.
        rc_r = ops.pair_cutoff(self.rc_r, self.atom_types, self.ib, self.jb)
        rc_a = ops.pair_cutoff(self.rc_a, self.atom_types, self.ib, self.jb)
        rc_r = rc_r.unsqueeze(-1) if torch.is_tensor(rc_r) else rc_r
        rc_a = rc_a.unsqueeze(-1) if torch.is_tensor(rc_a) else rc_a
        r_rad = torch.where(d < rc_r, rij, rij * (rc_r / d)).detach().requires_grad_(True)
        r_ang = torch.where(d < rc_a, rij, rij * (rc_a / d)).detach().requires_grad_(True)
        pi_r = pi_a = self.ib
        pj_r = pj_a = self.jb
        with torch.enable_grad():
            q = ops.compute_descriptors(
                r_rad, r_ang, pi_r, pj_r, pi_a, pj_a, self.atom_types, self.n, c.c2, c.c3, self.rc_r, self.rc_a,
                c.basis_size_radial, c.basis_size_angular, c.n_max_radial, c.n_max_angular, c.l_max_3b,
                c.has_q_222, c.has_q_1111, c.has_q_112, c.num_lm, c._c3b, c._c4b, c._c5b, c._c4b2,
                c.dtype, c.device, backend=self.backend, has_q_123=c.has_q_123, has_q_233=c.has_q_233,
                has_q_134=c.has_q_134)
            ei = ops.apply_ann(q * c.q_scaler, self.atom_types, c.num_types, c.w0, c.b0, c.w1, c.b1,
                               c.dtype, c.device)
            g_rad, g_ang = torch.autograd.grad(ei.sum(), [r_rad, r_ang], allow_unused=True)
        g_rad = torch.zeros_like(r_rad) if g_rad is None else g_rad
        g_ang = torch.zeros_like(r_ang) if g_ang is None else g_ang
        forces, _ = ops.accumulate_forces_virial(self.n, pi_r, pj_r, r_rad.detach(), g_rad, pi_a, pj_a,
                                                 r_ang.detach(), g_ang, c.dtype, c.device)
        return ei.detach().sum().reshape(1), forces


def make_student(path: str, device: str) -> NEPStudent:
    if NEPCalculator(path, dtype=torch.float32, device="cpu").has_zbl:
        raise NotImplementedError("a NEP with ZBL needs ops.compute_zbl added here")
    return NEPStudent(path, device)
