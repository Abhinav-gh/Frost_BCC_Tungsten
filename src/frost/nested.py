"""Exact teacher statistics from student-driven proposals (note section 5.3, Prop. 3).

The dynamics contract certifies a trajectory, and it cannot help a system whose
fast motion outruns any affordable student: water's hydrogens carry a broadband
~4% student error on forces that change by more than their own RMS within a
femtosecond (plan section 1.13). Many workloads need equilibrium averages, not
trajectories, and for those the student's error need not bias anything.

``NestedHMC`` is generalised hybrid Monte Carlo with a surrogate integrator:

1. partially refresh the momenta, v <- cos(phi) v + sin(phi) sigma xi;
2. run ``n_draft`` velocity-Verlet steps under the **student** force;
3. accept with min(1, exp(-(H_T(x', v') - H_T(x, v)) / kT)), where
   H_T = U_T + K uses the **teacher** energy, from one energy-only teacher call;
4. on rejection restore the state and reverse the momenta.

Velocity Verlet under any smooth force is volume preserving and time reversible,
so this is a valid Metropolis-Hastings step for exp(-H_T / kT) (Duane et al. 1987;
Gelb 2003; Coe et al. 2009). The student's error changes only the acceptance rate,
never the distribution. With ``refresh = 1`` this is plain HMC; smaller values keep
part of the dynamics between proposals (Horowitz's GHMC).

Two kernel changes address what a rejection costs (plan section 1.14):

- **Extra chances** (Sohl-Dickstein, Mudigonda and DeWeese 2014; Campos and
  Sanz-Serna 2015). A rejected draft is not thrown away: the student keeps drafting
  for another ``n_draft`` steps and the teacher tests again, up to
  ``extra_chances`` times. Chance k is accepted with

      P_k(z_0) = min(1 - sum_{j<k} P_j(z_0),
                     exp(-(H_k - H_0) / kT) (1 - sum_{j<k} P_j(F z_k)))

  where the second sum runs over the reverse chain from the momentum-flipped end,
  which visits z_{k-1}, ..., z_1. Then pi(z_0) P_k(z_0) = pi(z_k) P_k(F z_k): the
  kernel keeps exp(-H_T / kT) exactly. Only when every chance fails are the
  momenta flipped, which is what makes partial refresh usable at modest acceptance.
- **Random draft length** (Mackenzie 1989; Bou-Rabee and Sanz-Serna 2017). A fixed
  length near half a vibrational period sends those modes back where they started.
  ``jitter`` draws each proposal's length uniformly from n_draft * [1 - jitter,
  1 + jitter], independently of the state, so each length is a valid kernel and so
  is their mixture.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from ase import units

from frost.integrators import ForceFn, Integrator, MDState


@dataclass
class NestedHMC:
    masses: torch.Tensor  # (N, 1), amu
    temperature: float  # K
    dt_fs: float
    n_draft: int
    student_force: ForceFn
    teacher_energy: Callable[[torch.Tensor], torch.Tensor]
    refresh: float = 1.0  # 1.0: full momentum refresh (HMC); < 1 keeps some dynamics
    extra_chances: int = 0  # further drafts tested after a rejection, before flipping
    jitter: float = 0.0  # draft length drawn from n_draft * [1 - jitter, 1 + jitter]
    seed: int = 0
    accepted: int = 0
    proposed: int = 0
    history: list = field(default_factory=list)

    def __post_init__(self):
        self.kT = units.kB * self.temperature
        self.sigma = torch.sqrt(self.kT / self.masses)
        self.gen = torch.Generator(device=self.masses.device).manual_seed(self.seed)
        phi = 0.5 * math.pi * self.refresh
        self.c, self.s = math.cos(phi), math.sin(phi)

    def kinetic(self, v: torch.Tensor) -> float:
        return float(0.5 * (self.masses * v * v).sum())

    def draw(self, like: torch.Tensor) -> torch.Tensor:
        return self.sigma * torch.randn(like.shape, generator=self.gen, device=like.device,
                                        dtype=like.dtype)

    def draft_length(self) -> int:
        if self.jitter <= 0.0:
            return self.n_draft
        lo = max(1, round(self.n_draft * (1.0 - self.jitter)))
        hi = max(lo, round(self.n_draft * (1.0 + self.jitter)))
        return int(torch.randint(lo, hi + 1, (), generator=self.gen, device=self.masses.device))

    def chance_probabilities(self, h: list[float]) -> list[float]:
        """P_1..P_m of accepting at each chance of the chain z_0 -> z_1 -> ... -> z_m.

        ``h`` holds the teacher Hamiltonians H_0..H_m. The reverse chain from the
        flipped z_k has energies H_k, H_{k-1}, ..., H_1, so its probabilities come from
        the same function; the recursion is O(m^2) calls deep, trivial for m <= 4.
        """
        probs: list[float] = []
        for k in range(1, len(h)):
            forward_left = 1.0 - sum(probs)
            reverse_left = 1.0 - sum(self.chance_probabilities(h[k:0:-1]))
            if reverse_left <= 0.0 or forward_left <= 0.0:
                probs.append(0.0)
                continue
            # exp(-(H_k - H_0)/kT) * reverse_left in log space; anything above 1 is
            # clamped by forward_left <= 1 anyway, so capping at 0 cannot change the min
            log_bound = -(h[k] - h[0]) / self.kT + math.log(reverse_left)
            probs.append(min(forward_left, math.exp(min(log_bound, 0.0))))
        return probs

    def step(self, x: torch.Tensor, v: torch.Tensor, energy_t: float, f_student: torch.Tensor):
        """One proposal. Returns (x, v, teacher energy, student force at x, accepted).

        ``history`` records, per proposal, dH_T at the first chance and its two parts:
        dH_S, the student integrator's own energy error, and the change of the energy
        residual U_T - U_S. It also records how many chances were drafted.
        """
        if getattr(self, "_e_student", None) is None:
            self._e_student = float(self.student_force(x)[0])
        v = self.c * v + self.s * self.draw(v)
        k0 = self.kinetic(v)
        h = [energy_t + k0]
        length = self.draft_length()
        u = float(torch.rand((), generator=self.gen, device=x.device))

        state = MDState(x.clone(), v.clone(), self.masses)
        integ = Integrator(state, self.dt_fs)
        f = f_student
        cumulative = 0.0
        self.proposed += 1
        for chance in range(1 + self.extra_chances):
            e_s1 = self._e_student
            for _ in range(length):
                e_s1, f = integ.inner_step(self.student_force, f)
            e_s1 = float(e_s1)
            e1 = float(self.teacher_energy(state.x))
            k1 = self.kinetic(state.v)
            h.append(e1 + k1)
            if chance == 0:
                record = {
                    "dH_T": h[1] - h[0],
                    "dH_S": (e_s1 + k1) - (self._e_student + k0),
                    "d_residual": (e1 - e_s1) - (energy_t - self._e_student),
                }
            cumulative += self.chance_probabilities(h)[-1]
            if u < cumulative:
                record.update(accepted=True, chances=chance + 1, draft_fs=length * self.dt_fs)
                self.history.append(record)
                self.accepted += 1
                self._e_student = e_s1
                return state.x, state.v, e1, f, True
        record.update(accepted=False, chances=1 + self.extra_chances, draft_fs=length * self.dt_fs)
        self.history.append(record)
        return x, -v, energy_t, f_student, False

    @property
    def acceptance(self) -> float:
        return self.accepted / max(self.proposed, 1)
