"""Resonance-free multiple-time-step sampling on the teacher-student split: SIN(R).

Why this exists (plan section 1.15). Every multirate result so far is capped by
resonance: an impulse applied every H femtoseconds pumps the fast modes once H
passes about 0.2 of the fastest period, so a teacher call buys only 3-9 inner
student steps at matched accuracy. For trajectories that limit is essentially
fundamental to impulse splitting. For *sampling* it is not. Isokinetic
multiple-time-step integrators couple every degree of freedom to its own
thermostat through a kinetic-energy constraint, so no mode can accumulate the
energy a resonance would pump into it, and outer steps of 60-100 fs become usable
(Minary, Tuckerman & Martyna, PRL 93, 150201, 2004; Leimkuhler, Margul &
Tuckerman, Mol. Phys. 111, 3579, 2013, arXiv:1307.1167 -- "SIN(R)").

What it samples. The canonical distribution over *configurations*; velocities are
microcanonical per degree of freedom, not Maxwell-Boltzmann, and the dynamics are
deliberately unphysical. So this belongs to the statistics contract, judged on
configurational bias against a reference and on effective samples per second,
never on NVE drift, spectra or diffusion.

Equations (per degree of freedom alpha, with L thermostat pairs; paper Eqs. 10-12):

    dq  = v dt
    dv  = [F/m - lambda v] dt
    dv1_k = -lambda v1_k dt - v2_k v1_k dt
    dv2_k = [(Q1 v1_k^2 - kT)/Q2] dt - gamma v2_k dt + sigma dW_k

under the isokinetic constraint  m v^2 + L/(L+1) sum_k Q1 v1_k^2 = L kT.

Integrator: paper Eqs. 50 (XO-RESPA) and 52 (XI-RESPA). Each inner step is a half
kick, half drift, an exact Ornstein-Uhlenbeck step on v2, half drift, half kick; the
slow force enters as n * F_s in the first and last half kicks, which is an impulse
of (Delta t / 2) F_s at each end -- the same placement as r-RESPA, but through the
isokinetic kick, which cannot overshoot. The schemes differ only in where the
Nose-Hoover part goes. XO wraps the whole outer step with it, integrated over
Delta t / 2 in one Suzuki-Yoshida pass; XI wraps every inner step, over delta t / 2.
XI is the default because XO is under-resolved once Delta t exceeds the thermostat
time sqrt(Q1 / kT): on the paper's own quartic oscillator XO samples <q^2> correctly
to 2 pi/omega, is 24% off at 4 pi/omega and overflows at 8 pi/omega. The paper's
99 fs water runs used XI.

The isokinetic kick under a constant force F has an exact solution (Eqs. 53-62):
with a = F v / (L kT) and b = F^2 / (m L kT),

    s(t)  = sinh(sqrt(b) t)/sqrt(b) + (a/b)(cosh(sqrt(b) t) - 1)
    s'(t) = cosh(sqrt(b) t) + (a/sqrt(b)) sinh(sqrt(b) t)
    v  <- (v + (F/m) s(t)) / s'(t),     v1_k <- v1_k / s'(t).

The arXiv text prints s' with sinh and cosh exchanged, which cannot be right since
s'(0) must be 1; the form above solves s'' = b s + a with s(0) = 0, s'(0) = 1 and
satisfies s'^2 = 1 + 2 a s + b s^2, which is exactly the condition for the kick to
preserve the constraint. The unit tests check that preservation to rounding.
|a/sqrt(b)| <= 1 under the constraint, so s' >= exp(-sqrt(b) t) > 0 always.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from ase import units

ForceFn = Callable[[torch.Tensor], torch.Tensor]


def _sinhc(x: torch.Tensor) -> torch.Tensor:
    """sinh(x)/x, stable at small x."""
    small = x.abs() < 1e-3
    x2 = x * x
    series = 1.0 + x2 / 6.0 + x2 * x2 / 120.0
    safe = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, series, torch.sinh(safe) / safe)


def _coshm1_over_x2(x: torch.Tensor) -> torch.Tensor:
    """(cosh(x) - 1)/x^2, stable at small x."""
    small = x.abs() < 1e-3
    x2 = x * x
    series = 0.5 + x2 / 24.0 + x2 * x2 / 720.0
    safe = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, series, (torch.cosh(safe) - 1.0) / (safe * safe))


@dataclass
class SINR:
    """Stochastic isokinetic Nose-Hoover RESPA, vectorised over every degree of freedom.

    All quantities are in one consistent unit system chosen by the caller (energy,
    length, mass, time). ``from_ase`` builds one in ASE units from a temperature in
    kelvin and times in femtoseconds.
    """

    masses: torch.Tensor  # (N, 1)
    kT: float  # noqa: N815 -- energy; the physics name is clearer than kt
    dt: float  # inner step, time
    n_inner: int  # inner steps per outer step
    L: int = 4
    Q1: float = 1.0  # energy * time^2
    Q2: float = 1.0
    gamma: float = 1.0  # 1/time, friction on v2
    n_sy: int = 3  # Suzuki-Yoshida order for the Nose part
    n_res: int = 1
    seed: int = 0
    scheme: str = "XI"  # where the Nose part goes: every inner step (XI) or the outer (XO)
    x: torch.Tensor | None = None
    v: torch.Tensor | None = None
    v1: torch.Tensor | None = None  # (L, N, 3)
    v2: torch.Tensor | None = None  # (L, N, 3)
    history: list = field(default_factory=list)

    def __post_init__(self):
        self.Lam = self.L * self.kT
        self.c = self.L / (self.L + 1.0)
        self.gen = torch.Generator(device=self.masses.device).manual_seed(self.seed)
        if self.n_sy == 3:
            w1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
            self.weights = (w1, 1.0 - 2.0 * w1, w1)
        elif self.n_sy == 1:
            self.weights = (1.0,)
        else:
            raise ValueError("n_sy must be 1 or 3")
        if self.scheme not in ("XI", "XO"):
            raise ValueError("scheme must be 'XI' or 'XO'")

    @classmethod
    def from_ase(cls, masses: torch.Tensor, temperature_K: float, dt_fs: float, n_inner: int,
                 L: int = 4, tau_fs: float = 10.0, gamma_per_fs: float = 0.1, seed: int = 0):
        """SIN(R) in ASE units, with the paper's recommended Q1 = Q2 = kT tau^2."""
        kT = units.kB * temperature_K
        tau = tau_fs * units.fs
        return cls(masses=masses, kT=kT, dt=dt_fs * units.fs, n_inner=n_inner, L=L,
                   Q1=kT * tau * tau, Q2=kT * tau * tau, gamma=gamma_per_fs / units.fs, seed=seed)

    # --------------------------------------------------------------- state
    def start(self, x: torch.Tensor) -> None:
        """Initial velocities on the constraint surface, with every v1 positive.

        v takes a bounded fraction of the per-degree-of-freedom budget L kT and the
        remainder is shared equally among the L v1 variables; v2 starts from its
        stationary Gaussian. The dynamics forget this within a few tau.
        """
        m = self.masses.expand_as(x)
        z = torch.randn(x.shape, generator=self.gen, device=x.device, dtype=x.dtype)
        self.x = x.clone()
        self.v = 0.5 * torch.sqrt(self.Lam / m) * torch.tanh(z)
        rest = self.Lam - m * self.v * self.v
        share = rest * (self.L + 1.0) / (self.L * self.L * self.Q1)
        self.v1 = torch.sqrt(share).unsqueeze(0).expand(self.L, *x.shape).clone()
        self.v2 = math.sqrt(self.kT / self.Q2) * torch.randn(
            (self.L, *x.shape), generator=self.gen, device=x.device, dtype=x.dtype)

    def constraint_residual(self) -> float:
        """max |m v^2 + L/(L+1) sum Q1 v1^2 - L kT| / (L kT): zero up to rounding."""
        m = self.masses.expand_as(self.v)
        total = m * self.v * self.v + self.c * self.Q1 * (self.v1 * self.v1).sum(dim=0)
        return float(((total - self.Lam).abs() / self.Lam).max())

    # ----------------------------------------------------------- operators
    def _kick(self, force: torch.Tensor, t: float) -> None:
        """exp(iL_v t) under a constant force: exact, constraint-preserving."""
        m = self.masses.expand_as(self.v)
        a = force * self.v / self.Lam
        b = force * force / (m * self.Lam)
        root = torch.sqrt(b)
        xarg = root * t
        s = t * _sinhc(xarg) + a * t * t * _coshm1_over_x2(xarg)
        sdot = torch.cosh(xarg) + a * t * _sinhc(xarg)
        self.v = (self.v + force / m * s) / sdot
        self.v1 = self.v1 / sdot.unsqueeze(0)

    def _nose(self, t: float) -> None:
        """exp(iL_N t), Suzuki-Yoshida split, each piece solved exactly (Eqs. 63-70)."""
        m = self.masses.expand_as(self.v)
        for w in self.weights:
            for _ in range(self.n_res):
                d = w * t / self.n_res
                self.v2 = self.v2 + (self.Q1 * self.v1 * self.v1 - self.kT) / self.Q2 * (0.5 * d)
                decay = torch.exp(-self.v2 * d)
                denom = m * self.v * self.v + self.c * self.Q1 * (self.v1 * self.v1 * decay * decay).sum(0)
                h = torch.sqrt(self.Lam / denom)
                self.v = self.v * h
                self.v1 = self.v1 * h.unsqueeze(0) * decay
                self.v2 = self.v2 + (self.Q1 * self.v1 * self.v1 - self.kT) / self.Q2 * (0.5 * d)

    def _ou(self, t: float) -> None:
        """exp(iL_OU t) on v2: exact Ornstein-Uhlenbeck step (Eq. 51)."""
        c = math.exp(-self.gamma * t)
        s = math.sqrt((1.0 - c * c) * self.kT / self.Q2)
        noise = torch.randn(self.v2.shape, generator=self.gen, device=self.v2.device, dtype=self.v2.dtype)
        self.v2 = c * self.v2 + s * noise

    # ---------------------------------------------------------------- step
    def outer_step(self, fast: ForceFn, f_fast: torch.Tensor, slow_now: torch.Tensor,
                   slow_at: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        """One outer step of n_inner inner steps (paper Eq. 52 for XI, Eq. 50 for XO).

        ``fast(x)`` returns the fast force (the student); ``slow_now`` is the slow
        force (the residual F_T - F_S) at the current positions; ``slow_at(x, f)``
        evaluates it at the new positions given the fast force already computed
        there, which is the one teacher call the outer step makes. Returns the fast
        and slow forces at the new positions, ready for the next step.
        """
        n, h = self.n_inner, self.dt
        inner = self.scheme == "XI"
        if not inner:
            self._nose(0.5 * n * h)
        slow_new = slow_now
        for i in range(n):
            if inner:
                self._nose(0.5 * h)
            self._kick(f_fast + (n * slow_now if i == 0 else 0.0), 0.5 * h)
            self.x = self.x + 0.5 * h * self.v
            self._ou(h)
            self.x = self.x + 0.5 * h * self.v
            f_fast = fast(self.x)
            if i == n - 1:
                slow_new = slow_at(self.x, f_fast)
            self._kick(f_fast + (n * slow_new if i == n - 1 else 0.0), 0.5 * h)
            if inner:
                self._nose(0.5 * h)
        if not inner:
            self._nose(0.5 * n * h)
        return f_fast, slow_new
