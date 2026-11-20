"""Path-integral MD with ring-polymer contraction, the core of Frost's imaginary-time axis (further_expansions_beyond.md 6.6).

A path-integral simulation runs P copies (beads) of the system joined by harmonic springs into a ring; its statistics are
those of quantum nuclei (Feynman's path integral). Each bead feels the physical potential V(q_j) and springs
1/2 m omega_P^2 (q_j - q_{j+1})^2 with omega_P = P k_B T / hbar, and the ring is sampled at P T (i-PI's convention).

What this module provides, all on GPU tensors (beads, atoms, 3) in ASE units:
- ``NormalModes``: the real orthonormal transform of the free ring (Ceriotti et al. 2010), frequencies
  omega_k = 2 omega_P sin(k pi / P), the free ring propagated exactly in normal modes, and the PILE-L thermostat
  (critical damping gamma_k = 2 omega_k on the internal modes, gamma_0 = 1 / tau_0 on the centroid; Ceriotti,
  Parrinello, Markland and Manolopoulos, JCP 133, 124104, 2010).
- ``Contraction``: ring-polymer contraction (Markland and Manolopoulos, JCP 129, 024105, 2008) from P to P' beads by
  trigonometric interpolation keeping the P' lowest ring modes; a force on the contracted beads maps back by
  F = (P / P') T^T F', which keeps the centroid and returns a uniform force unchanged.
- ``RingPolymer``: the state, a BAOAB-style step (B: physical kick, A: free-ring propagation, O: PILE), and the
  centroid-virial quantum kinetic energy.
Frost's multirate structure sits on top (``pimd_runtime``): the student on every bead every step, the teacher on the
contracted ring every k steps, the held residual expanded from the contracted beads.
"""

from __future__ import annotations

import math

import torch
from ase import units

HBAR = units._hbar * units.J * units.s  # eV x ASE time


class NormalModes:
    """Real normal modes of a free ring of P beads, ordered 0 (centroid), cos 1..P/2-ish, sin ..., as i-PI does."""

    def __init__(self, n_beads: int, temperature_K: float, device="cpu", dtype=torch.float64):
        p = n_beads
        self.P = p
        self.kT_P = p * units.kB * temperature_K  # the ring is sampled at P T
        self.omega_p = self.kT_P / HBAR
        c = torch.zeros(p, p, dtype=dtype)
        j = torch.arange(p, dtype=dtype)
        for k in range(p):
            if k == 0:
                c[:, k] = math.sqrt(1.0 / p)
            elif k < p / 2:
                c[:, k] = math.sqrt(2.0 / p) * torch.cos(2 * math.pi * j * k / p)
            elif k == p / 2:
                c[:, k] = math.sqrt(1.0 / p) * (-1.0) ** j
            else:
                c[:, k] = math.sqrt(2.0 / p) * torch.sin(2 * math.pi * j * k / p)
        self.C = c.to(device)  # q_bead = C q_mode, q_mode = C^T q_bead
        self.omega = (2.0 * self.omega_p * torch.sin(torch.arange(p, dtype=dtype) * math.pi / p)).to(device)

    def to_modes(self, q: torch.Tensor) -> torch.Tensor:
        return torch.einsum("jk,jnd->knd", self.C.to(q.dtype), q)

    def to_beads(self, qm: torch.Tensor) -> torch.Tensor:
        return torch.einsum("jk,knd->jnd", self.C.to(qm.dtype), qm)

    def free_ring(self, q: torch.Tensor, v: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Propagate the free ring (springs only) exactly for dt; v are velocities (A / ASE time)."""
        qm, vm = self.to_modes(q), self.to_modes(v)
        w = self.omega.to(q.dtype).view(-1, 1, 1)
        cos, sin = torch.cos(w * dt), torch.sin(w * dt)
        safe = torch.where(w > 0, w, torch.ones_like(w))
        q_new = torch.where(w > 0, cos * qm + sin / safe * vm, qm + dt * vm)
        v_new = torch.where(w > 0, -w * sin * qm + cos * vm, vm)
        return self.to_beads(q_new), self.to_beads(v_new)


class PILE:
    """PILE-L: Langevin in normal modes, critical damping on internal modes, a gentle centroid friction."""

    def __init__(self, modes: NormalModes, masses: torch.Tensor, dt: float, tau0_fs: float = 100.0, seed: int = 0):
        g = 2.0 * modes.omega.clone()
        g[0] = 1.0 / (tau0_fs * units.fs)
        self.c1 = torch.exp(-g * dt).view(-1, 1, 1)
        self.c2 = torch.sqrt(1.0 - self.c1 ** 2)
        self.sigma = torch.sqrt(modes.kT_P / masses).unsqueeze(0)  # (1, N, 1): thermal velocity at P T
        self.modes = modes
        self.gen = torch.Generator(device=masses.device).manual_seed(seed)

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        vm = self.modes.to_modes(v)
        noise = torch.randn(vm.shape, generator=self.gen, device=v.device, dtype=v.dtype)
        vm = self.c1.to(v.dtype) * vm + self.c2.to(v.dtype) * self.sigma.to(v.dtype) * noise
        return self.modes.to_beads(vm)


class Contraction:
    """Map P beads to P' beads keeping the P' lowest ring modes (trigonometric interpolation), and forces back."""

    def __init__(self, n_beads: int, n_contracted: int, device="cpu", dtype=torch.float64):
        p, pc = n_beads, n_contracted
        if pc > p:
            raise ValueError("cannot contract to more beads")
        self.P, self.Pc = p, pc
        # T (P' x P) built by applying the map to unit vectors: FFT over P, keep |w| < P'/2 (and w = P'/2 combined for
        # even P'), inverse FFT over P'
        eye = torch.eye(p, dtype=torch.complex128)
        qh = torch.fft.fft(eye, dim=0)  # (P freq, P basis)
        qh_c = torch.zeros(pc, p, dtype=torch.complex128)
        half = pc // 2
        for w in range(-((pc - 1) // 2), (pc - 1) // 2 + 1):
            qh_c[w % pc] = qh[w % p] * (pc / p)
        if pc % 2 == 0:
            # the P' Nyquist mode is the cosine at w = P'/2: a real path's +-P'/2 components add on the P' samples
            # (their sine part vanishes there), so it keeps both; with P' = P it is the single Nyquist component
            qh_c[half] = qh[half] if p == pc else (qh[half] + qh[p - half]) * (pc / p)
        t = torch.fft.ifft(qh_c, dim=0).real
        self.T = t.to(dtype).to(device)  # (P', P)

    def contract(self, q: torch.Tensor) -> torch.Tensor:
        return torch.einsum("cj,jnd->cnd", self.T.to(q.dtype), q)

    def expand_force(self, f: torch.Tensor) -> torch.Tensor:
        """Force on the full ring from forces on the contracted beads: (P / P') T^T F'."""
        return (self.P / self.Pc) * torch.einsum("cj,cnd->jnd", self.T.to(f.dtype), f)


class RingPolymer:
    """P beads of N atoms; one step: B (physical kick, dt/2), A (free ring, dt/2), O (PILE), A (dt/2), B (dt/2)."""

    def __init__(self, q: torch.Tensor, v: torch.Tensor, masses: torch.Tensor, temperature_K: float, dt_fs: float,
                 tau0_fs: float = 100.0, seed: int = 0):
        self.q, self.v, self.masses = q, v, masses  # (P, N, 3), (P, N, 3), (N, 1)
        self.P = q.shape[0]
        self.dt = dt_fs * units.fs
        self.temperature = temperature_K
        self.modes = NormalModes(self.P, temperature_K, device=q.device, dtype=torch.float64)
        self.pile = PILE(self.modes, masses, self.dt, tau0_fs, seed)

    def kick(self, force: torch.Tensor, duration: float) -> None:
        self.v = self.v + duration * force.to(self.v.dtype) / self.masses.unsqueeze(0)

    def step(self, force_fn, f: torch.Tensor):
        """force_fn(q (P, N, 3)) -> (energies (P,), forces (P, N, 3)); f: forces at the current q. Returns (e, f)."""
        h = self.dt
        self.kick(f, 0.5 * h)
        self.q, self.v = self.modes.free_ring(self.q, self.v, 0.5 * h)
        self.v = self.pile(self.v)
        self.q, self.v = self.modes.free_ring(self.q, self.v, 0.5 * h)
        e, f = force_fn(self.q)
        self.kick(f, 0.5 * h)
        return e, f

    def centroid(self) -> torch.Tensor:
        return self.q.mean(0)

    def kinetic_cv(self, f: torch.Tensor) -> float:
        """Centroid-virial quantum kinetic energy (eV): 3 N k_B T / 2 - 1/(2P) sum_j (q_j - q_c) . F_j."""
        n = self.q.shape[1]
        dq = self.q - self.centroid().unsqueeze(0)
        return 1.5 * n * units.kB * self.temperature - 0.5 / self.P * float((dq * f).sum())


def fit_ring_curvature(dr: torch.Tensor, dq: torch.Tensor, ridge: float = 1e-6) -> tuple[torch.Tensor, float]:
    """Each atom's symmetric 3x3 J_i with dr_b,i ~ J_i dq_b,i over the beads b (2026-11-17, the ring hold).

    dr: (P, N, 3) the residual's deviation from its bead mean; dq: (P, N, 3) the beads' offsets from their centroid.
    Least squares per atom, J_i = (sum_b dr dq^T)(sum_b dq dq^T + ridge)^-1, then symmetrised so the force J_i dq is the
    gradient of a potential. Returns (J (N, 3, 3), the fraction of the variation of dr it explains)."""
    a = torch.einsum("bni,bnj->nij", dr, dq)
    b = torch.einsum("bni,bnj->nij", dq, dq)
    lam = ridge * b.diagonal(dim1=1, dim2=2).mean(-1).clamp_min(1e-12)
    j = a @ torch.linalg.inv(b + lam[:, None, None] * torch.eye(3, dtype=b.dtype, device=b.device))
    j = 0.5 * (j + j.transpose(1, 2))
    miss = dr - torch.einsum("nij,bnj->bni", j, dq)
    return j, float(1.0 - miss.pow(2).sum() / dr.pow(2).sum().clamp_min(1e-30))

