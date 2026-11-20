"""The certificate as clauses: priors that bound an interval, checks on each closed interval.

A clause has up to two hooks:
- ``cap(context) -> int | None``: the largest interval it admits before anything runs (a prior);
- ``check(interval) -> str | None``: a reason to reject a closed interval, or None.

and a ``kind`` deciding what a rejection does:
- "rollback": the interval is undone and redone as two halves, down to k = 1 (the error is local
  to this interval: magnitude, non-finite state, a CFD hold error);
- "refuse": the interval is latched off for the rest of the run (the error accumulates and only a
  measurement over many intervals sees it: drift, a regime envelope).

The two monitors from approximate computing are clauses as well, so the comparison runs through
the same executor: ``QoSThreshold`` (Green: back off when the sampled output error exceeds a
threshold) and ``WorstCase`` (Rumba: re-execute when one output's error is large).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from frost.controller import DriftMonitor, IntervalController


class RunningFit:
    """Least-squares slope of (t, e) and its standard error over a sliding window, in O(1) per point.

    The bias clause refitted its whole 10 ps window in Python every interval: at water's k = 2 that is 10,000 refits
    of up to 5,000 points per 20 ps, which cost the certified arm 1.34 ms/fs against the same k uncertified (1.08x
    against 1.40x). Running sums give the same fit (DriftMonitor._fit's formulas) at constant cost. Times are
    centred on the first point so the sums stay well conditioned.
    """

    def __init__(self):
        self.points = deque()
        self.t0 = None
        self.n = 0
        self.st = self.se = self.stt = self.ste = self.see = 0.0

    def _add(self, t, e, sign):
        x = t - self.t0
        self.n += sign
        self.st += sign * x
        self.se += sign * e
        self.stt += sign * x * x
        self.ste += sign * x * e
        self.see += sign * e * e

    def append(self, t: float, e: float) -> None:
        if self.points and self.points[-1][0] == t:  # idempotent on a repeated time (DriftMonitor._append)
            self._add(*self.points.pop(), -1)
        if self.t0 is None:
            self.t0 = t
        self.points.append((t, e))
        self._add(t, e, 1)

    def trim(self, window: float) -> None:
        while len(self.points) > 2 and self.points[-1][0] - self.points[0][0] > window:
            self._add(*self.points.popleft(), -1)

    def span(self) -> float:
        return self.points[-1][0] - self.points[0][0] if self.points else 0.0

    def fit(self) -> tuple[float, float] | None:
        n = self.n
        if n < 8:
            return None
        var = self.stt - self.st * self.st / n
        if var <= 0.0:
            return None
        slope = (self.ste - self.st * self.se / n) / var
        intercept = (self.se - slope * self.st) / n
        resid = (self.see - 2 * intercept * self.se - 2 * slope * self.ste + n * intercept * intercept
                 + 2 * intercept * slope * self.st + slope * slope * self.stt)
        return slope, math.sqrt(max(resid, 0.0) / ((n - 2) * var)) if n > 2 else float("inf")

    def clear(self) -> None:
        self.__init__()


class Clause:
    name = "clause"
    kind = "rollback"

    def cap(self, context: dict) -> int | None:
        return None

    def check(self, iv) -> str | None:
        return None

    def reset(self) -> None:
        pass

    def resume(self) -> None:
        """The executor left the never-lose baseline for the multirate path again. Default: judge afresh."""
        self.reset()


@dataclass
class Magnitude(Clause):
    """The held quantity's change over one interval stays under epsilon (MD: 2.0 meV/(A amu)).

    ``scale`` (optional, a callable with no arguments) multiplies epsilon at each check: ``md_runtime.py
    --thermal-epsilon`` passes sqrt(T_target / T_start), so a bound set at the start temperature keeps its meaning when
    the thermostat's target moves (thermal force fluctuations grow as sqrt(T)). The target, not the measured
    temperature, so an interval that heats cannot loosen its own bound."""
    epsilon: float = 2.0
    name: str = "magnitude"
    kind: str = "rollback"
    scale: object = None
    # Small-system allowance (opt-in, md_runtime --small-n-allowance, 2026-11-15): one interval's staleness is an RMS
    # over 3N components, with relative scatter ~1 / sqrt(6 N), so a hard per-interval threshold calibrated at ~500 atoms
    # fails on noise alone in a small molecule (ethanol, 9 atoms, k 2 at 0.83 of epsilon on average, 23% of intervals
    # over it). With n_atoms set, the per-interval threshold carries the region clause's allowance, normalised to the
    # calibration size: epsilon x (1 + z / sqrt(6 N)) / (1 + z / sqrt(6 n_ref)); unchanged at n_ref, 1.44 epsilon at N 9,
    # 0.95 epsilon at N 8,192. The controller still grows on the running mean against epsilon itself.
    n_atoms: int | None = None
    n_ref: int = 500
    z: float = 4.0

    def allowance(self) -> float:
        if not self.n_atoms:
            return 1.0
        return (1.0 + self.z / (6.0 * self.n_atoms) ** 0.5) / (1.0 + self.z / (6.0 * self.n_ref) ** 0.5)

    def check(self, iv):
        eps = self.epsilon * (float(self.scale()) if self.scale is not None else 1.0) * self.allowance()
        return None if iv.staleness <= eps else f"staleness {iv.staleness:.3f} > {eps:.3f}"


@dataclass
class LocalMagnitude(Magnitude):
    """The magnitude clause region by region (frost.locality, 2026-11-14): the worst region's RMS change, divided by its
    allowance 1 + z / sqrt(6 m) for m atoms, stays under epsilon. The whole-cell RMS dilutes a local event (a cascade's
    core, a defect, a reaction): 100 hot atoms among 8,192 can be 9x over before it moves, while the allowance keeps this
    clause silent wherever staleness is spread evenly. Reads ``iv.extra["local"]`` (MDBackend computes it when built with
    ``local=``); silent when it is absent."""
    name: str = "local magnitude"

    def check(self, iv):
        local = iv.extra.get("local")
        if local is None:
            return None
        eps = self.epsilon * (float(self.scale()) if self.scale is not None else 1.0)
        return None if local <= eps else f"worst region {local:.3f} > {eps:.3f}"


@dataclass
class NonFinite(Clause):
    name: str = "non-finite"
    kind: str = "rollback"

    def check(self, iv):
        return None if iv.finite else "non-finite state"


@dataclass
class ResonancePrior(Clause):
    """k dt <= 0.22 of the fastest period, even in femtoseconds (frost.controller).

    The lattice is the controller's: the executor passes its controller's fine lattice in ``context["fine_fs"]`` (every
    step count below fine_fs is a lattice point). Until 2026-11-14 this clause rebuilt the cap on the even lattice alone,
    so a run with ``--fine-lattice 4`` had a controller that allowed k 3 (the peptide in NVE: 0.22 x 14.59 fs = 3.2 fs)
    and an executor that silently capped it at 2: two computations of one bound, disagreeing."""
    period_fs: float = 0.0
    dt_fs: float = 1.0
    fraction: float = 0.22
    fine_fs: float | None = None  # None: the controller's (from the executor's context)
    name: str = "resonance"
    kind: str = "prior"

    def cap(self, context):
        if not self.period_fs:
            return None
        fine = self.fine_fs if self.fine_fs is not None else float(context.get("fine_fs", 0.0))
        return IntervalController(period_fs=self.period_fs, dt_fs=self.dt_fs, resonance_fraction=self.fraction,
                                  k_max=10**6, fine_fs=fine).resonance_cap()


@dataclass
class DriftPosterior(Clause):
    """NVE only: the energy gained since the interval was adopted, projected to 50 ps, < 1 meV/atom."""
    monitor: DriftMonitor = field(default_factory=DriftMonitor)
    name: str = "drift"
    kind: str = "refuse"

    on_admit: object = None  # sequential mode: called with (k, name) once the interval is judged inside half the budget
    k_now: int | None = None
    k_since_ps: float = 0.0
    rewind: bool = False  # the executor rewinds the refused interval's segment (Executor.segment)
    measured_step: bool = False  # with rewind: step down by the measured excess, not by half (step_factor)

    def step_factor(self):
        return self.monitor.step_factor()

    def rewind_to(self, t_fs: float) -> None:
        self.monitor.truncate(t_fs / 1e3)

    def check(self, iv):
        if iv.energy is None:
            return None
        if self.monitor.over_projection(iv.energy, iv.t):
            return "projected gain over budget"
        if self.on_admit is not None and self.monitor.sequential:
            if iv.k != self.k_now:  # admission judges an interval on its own record (ThermostatBias does the same)
                self.k_now, self.k_since_ps = iv.k, iv.t / 1e3 + self.monitor.block_ps
            if self.monitor.admits(self.k_since_ps):
                self.on_admit(iv.k, self.name)
        return None

    def reset(self):
        self.monitor.reset()
        self.k_now = None

    def resume(self):
        """Sequential mode keeps the record across a never-lose resume: it resumes at the interval that fell back
        (probe_k), and growth never resets the record either, so only a refusal starts a new one. Resetting here
        let Li3PO4 at 1000 K gain +3.24 meV/atom over 50 ps with no refusal: it fell back and resumed more often
        than every 20 ps, so no record ever reached min_ps (gaps.md E.9)."""
        if not self.monitor.sequential:
            self.monitor.reset()


class RunningFitCov:
    """Least-squares slope of e on t with a covariate u held out, e = a + b t + c u, over a sliding window, O(1) per point.

    Under a barostat the effective energy E - Q carries a term that follows the volume, not time: each box move books
    the student's exact energy change, and the residual's share, -(W_T - W_S) ln mu, is left in E - Q, where it moves
    with ln V and does not accumulate (a function of the volume, to first order). On Cu 500 the descriptor student's
    virial is ~1,500 eV off the teacher's (its training never varied the volume), so E - Q swings ~2.6 meV/atom with the
    volume and 5 ps slopes read +-0.5 meV/atom/ps of "shadow work" that is not there: the bias clause refused k 24 at
    +1.97 K predicted (gaps.md NPT results). With u = ln V regressed out, the same records give slopes of -0.005 to
    +0.023. The slope's standard error comes from the full 3 x 3 normal equations, so the covariate's cost in degrees of
    freedom and its correlation with time are both charged to it.
    """

    def __init__(self):
        self.points = deque()
        self.t0 = self.u0 = None
        self.s = np.zeros((3, 3))
        self.b = np.zeros(3)
        self.see = 0.0
        self.n = 0

    def _add(self, t, e, u, sign):
        z = np.array([1.0, t - self.t0, u - self.u0])
        self.s += sign * np.outer(z, z)
        self.b += sign * z * e
        self.see += sign * e * e
        self.n += sign

    def append(self, t: float, e: float, u: float) -> None:
        if self.points and self.points[-1][0] == t:
            self._add(*self.points.pop(), -1)
        if self.t0 is None:
            self.t0, self.u0 = t, u
        self.points.append((t, e, u))
        self._add(t, e, u, 1)

    def trim(self, window: float) -> None:
        while len(self.points) > 2 and self.points[-1][0] - self.points[0][0] > window:
            self._add(*self.points.popleft(), -1)

    def clear(self) -> None:
        self.__init__()

    def span(self) -> float:
        return self.points[-1][0] - self.points[0][0] if self.points else 0.0

    def fit(self) -> tuple[float, float] | None:
        if self.n < 12:
            return None
        try:
            inv = np.linalg.inv(self.s)
        except np.linalg.LinAlgError:
            return None
        beta = inv @ self.b
        resid = self.see - 2.0 * beta @ self.b + beta @ self.s @ beta
        var = max(resid, 0.0) / (self.n - 3)
        return float(beta[1]), math.sqrt(var * max(inv[1, 1], 0.0))


@dataclass
class ThermostatBias(Clause):
    """NVT: the temperature bias the integrator's own energy injection implies, under ``tolerance`` of T.

    Under a thermostat the energy an interval injects (its shadow work, the effective energy E - Q with the
    thermostat's heat Q taken out: ``Integrator.track_heat``) does not accumulate, so the NVE drift budget reads it
    wrongly in both directions. In steady state Langevin friction gamma removes energy at 2 gamma (K - K0), and with
    the injected power p per atom shared equally between kinetic and potential energy the balance gives the bias
    directly: dT = p / (3 gamma k_B). That is the observable an NVT run gets wrong, so the clause certifies it.
    p is the least-squares slope of the effective energy over a trailing ``window_ps``, judged once ``min_ps`` of it
    exists; a refusal needs the bias to clear the tolerance by two standard errors of the fit (DriftPosterior's
    rule), so noise alone does not refuse. Water at 1/ps: fixed 4 injects 0.85 meV/atom/ps, predicting +3.3 K
    against +2.6 K measured; fixed 5 1.58, predicting +6.1 against +4.0.
    """
    friction_per_ps: float = 1.0
    temperature_K: float = 300.0
    tolerance: float = 0.005  # relative: 1.5 K at 300 K
    window_ps: float = 10.0
    min_ps: float = 5.0
    name: str = "thermostat-bias"
    kind: str = "refuse"
    points: RunningFit = field(default_factory=RunningFit)
    on_admit: object = None  # called with (k, name) once the bias is judged under half the tolerance with margin
    k_points: RunningFit = field(default_factory=RunningFit)  # points at the current interval only (admission)
    k_now: int | None = None
    # Predictive admission (opt-in, md_runtime --predictive-admission, 2026-11-14): the admitted rung's measured margin
    # sets how far the next may go. With the bias growing as k^p, the rung whose bias would reach the admission bar is
    # k / rho^(1/p), rho = (|bias| + 2 SE) / (tolerance / 2); p = 3 sits above every growth measured so far (Si at 1000 K
    # 1.2-1.8, water near its cap 2.8), and the step is held to [1.25, 2]. The evidence per rung is unchanged (min_ps of
    # the rung's own data); only the step changes, and a rung that overshoots is refused and latched as before. Copper's
    # one-rung climb (24 -> 30 -> 38 -> 48 at 5 ps a rung) held its 20 ps average under fixed 48's (13.76x against 16.16x).
    predictive: bool = False
    admit_exponent: float = 3.0
    admit_max_factor: float = 2.0

    def bias(self, points: RunningFit | None = None) -> tuple[float, float] | None:
        """(predicted dT in K, its standard error) from the points so far, or None while too short."""
        points = self.points if points is None else points
        if points.span() < self.min_ps:
            return None
        fit = points.fit()
        if fit is None:
            return None
        per_k = 1e-3 / (3.0 * self.friction_per_ps * 8.617333262e-5)  # meV/atom/ps -> K
        return fit[0] * per_k, fit[1] * per_k

    def check(self, iv):
        if iv.energy is None:
            return None
        vol = iv.extra.get("volume") if iv.extra else None
        if vol is not None and not isinstance(self.points, RunningFitCov):
            # under a barostat: fit E - Q against time with ln V held out (RunningFitCov), from the first point on
            self.points, self.k_points = RunningFitCov(), RunningFitCov()
        cov = (math.log(vol),) if vol is not None else ()
        self.points.append(iv.t / 1e3, iv.energy, *cov)
        self.points.trim(self.window_ps)
        if iv.k != self.k_now:  # admission judges an interval only on its own data: the trailing window still
            self.k_now = iv.k  # holds the smaller k's points, which admitted every rung at once
            self.k_points.clear()
        self.k_points.append(iv.t / 1e3, iv.energy, *cov)
        limit = self.tolerance * self.temperature_K
        if self.on_admit is not None:
            bk = self.bias(self.k_points)
            if bk is not None and bk[0] + 2.0 * bk[1] < 0.5 * limit:
                if self.predictive:
                    rho = max(abs(bk[0]) + 2.0 * bk[1], 1e-6 * limit) / (0.5 * limit)
                    g = min(self.admit_max_factor, max(1.25, rho ** (-1.0 / self.admit_exponent)))
                    self.on_admit(iv.k, self.name, g)
                else:
                    self.on_admit(iv.k, self.name)
        b = self.bias()
        if b is None:
            return None
        return f"predicted bias {b[0]:+.2f} K > {limit:.2f} K" if b[0] - limit > 2.0 * b[1] else None

    def reset(self):
        self.points.clear()
        self.k_points.clear()
        self.k_now = None


@dataclass
class KineticBias(Clause):
    """NVT: the kinetic temperature itself, averaged over every inner step, against the thermostat's target.

    ``ThermostatBias`` sees what the integrator dissipates; it cannot see a bias that injects no net energy. A less
    accurate student kicks harder at the interval boundaries, and sampled over all phases of the interval its kinetic
    temperature could in principle sit above the target with no energy drift. This clause measures the observable
    directly (``MDBackend(track_temperature=True)`` puts each interval's step-averaged temperature in
    ``extra["T_mean"]``) and refuses when the trailing mean differs from the target by more than tolerance x T,
    by two standard errors of batch means (8 batches).

    **Noise-limited; opt-in only.** The mean kinetic temperature of N atoms over a window W scatters by about
    T sqrt(2/3N) sqrt(tau/W), tau ~ 0.5 ps at 1/ps friction: ~1.7 K for 512 atoms over 20 ps, more over the 10 ps
    window, and a test repeated every interval fires on noise. On MgO it refused the validated k = 10 at +7.16 K (the
    start-up transient) and k = 8 at -5.19 K, holding the run below its own fixed control. Use it only where the
    floor is well under the tolerance (thousands of atoms, long windows); ``ThermostatBias`` is the NVT certificate.
    """
    temperature_K: float = 300.0
    tolerance: float = 0.005
    window_ps: float = 10.0
    min_ps: float = 5.0
    batches: int = 8
    name: str = "kinetic-bias"
    kind: str = "refuse"
    samples: list = field(default_factory=list)  # (t ps, k, T_mean)
    on_admit: object = None  # called with (k, name) once |bias| is judged under half the tolerance with margin

    def bias(self) -> tuple[float, float] | None:
        """(mean T - target in K, its standard error), weighted by interval length, or None while too short."""
        if not self.samples or self.samples[-1][0] - self.samples[0][0] < self.min_ps:
            return None
        n = len(self.samples)
        if n < 2 * self.batches:
            return None
        size = n // self.batches
        means = []
        for b in range(self.batches):
            chunk = self.samples[b * size:(b + 1) * size]
            w = sum(k for _, k, _ in chunk)
            means.append(sum(k * t for _, k, t in chunk) / w)
        mean = sum(means) / len(means)
        var = sum((m - mean) ** 2 for m in means) / (len(means) - 1)
        return mean - self.temperature_K, (var / len(means)) ** 0.5

    def check(self, iv):
        t_mean = iv.extra.get("T_mean")
        if t_mean is None:
            return None
        self.samples.append((iv.t / 1e3, iv.k, float(t_mean)))
        while len(self.samples) > 2 and self.samples[-1][0] - self.samples[0][0] > self.window_ps:
            self.samples.pop(0)
        b = self.bias()
        if b is None:
            return None
        limit = self.tolerance * self.temperature_K
        if self.on_admit is not None and abs(b[0]) + 2.0 * b[1] < 0.5 * limit:
            self.on_admit(iv.k, self.name)
        return f"kinetic bias {b[0]:+.2f} K beyond {limit:.2f} K" if abs(b[0]) - limit > 2.0 * b[1] else None

    def reset(self):
        self.samples.clear()


@dataclass
class StructureClause(Clause):
    """Self-referenced structure: growth past the validated prior interval only if the pair distances do not move.

    Under a thermostat the bias clause certifies the temperature, and silica showed that is not enough: left to climb
    to k 9-12 on PET, its RDF moved 2.3x beyond the control's own window-to-window spread while T and the predicted bias
    stayed under 1 K (resonance moves energy among modes without heating the cell). Nothing else in the NVT certificate
    measures structure. This clause makes the run its own reference. While it runs at ``k_ref`` (the prior interval,
    0.22 of the fastest period, validated in NVE), after ``warm_ps``, it records ``ref_ps`` of the pair-distance
    histogram (``MDBackend(structure=...)``: the student list's distances, normalised, every few steps); the two halves
    of that record give its own spread. A larger interval is admitted (``on_admit``, for the probation) only once
    ``test_ps`` of its own data match the reference within that spread (the mean of its L1 distances to the two halves,
    windows of equal length, the fidelity gate's criterion), and refused when the distance passes ``refuse_factor``
    times the spread. A refusal does not discard the reference.
    """
    k_ref: int = 1
    warm_ps: float = 5.0
    ref_ps: float = 20.0
    test_ps: float = 10.0
    refuse_factor: float = 2.0
    name: str = "structure"
    kind: str = "refuse"
    on_admit: object = None
    ref: list = field(default_factory=list)  # (t ps, weighted histogram)
    ref_halves: tuple | None = None
    spread: float | None = None
    cur: list = field(default_factory=list)
    k_now: int | None = None
    last: dict = field(default_factory=dict)

    @staticmethod
    def _l1(a, b) -> float:
        return float(abs(b - a).sum() / max(abs(a).sum(), 1e-30))

    @staticmethod
    def _mean(rows):
        total = sum(h for _, h in rows)
        return total / max(total.sum(), 1e-30)

    def check(self, iv):
        h = iv.extra.get("pair_hist")
        if h is None:
            return None
        t = iv.t / 1e3
        if self.spread is None:
            if iv.k == self.k_ref and t >= self.warm_ps:
                self.ref.append((t, h * iv.k))
            if self.ref and self.ref[-1][0] - self.ref[0][0] >= self.ref_ps:
                mid = 0.5 * (self.ref[0][0] + self.ref[-1][0])
                a = self._mean([r for r in self.ref if r[0] < mid])
                b = self._mean([r for r in self.ref if r[0] >= mid])
                self.ref_halves, self.spread = (a, b), self._l1(a, b)
                self.last = {"t": t, "spread": self.spread}
            if self.spread is not None and self.on_admit is not None:
                self.on_admit(self.k_ref, self.name)
            return None
        if iv.k <= self.k_ref:
            if iv.k == self.k_ref and self.on_admit is not None:
                self.on_admit(iv.k, self.name)
            return None
        if iv.k != self.k_now:
            self.k_now, self.cur = iv.k, []
        self.cur.append((t, h * iv.k))
        if self.cur[-1][0] - self.cur[0][0] < self.test_ps:
            return None
        test = self._mean(self.cur)
        d = 0.5 * (self._l1(self.ref_halves[0], test) + self._l1(self.ref_halves[1], test))
        self.last = {"t": t, "k": iv.k, "distance": d, "spread": self.spread}
        if d <= self.spread and self.on_admit is not None:
            self.on_admit(iv.k, self.name)
        if d > self.refuse_factor * self.spread:
            return f"pair distances moved {d:.4f} > {self.refuse_factor} x their own spread {self.spread:.4f}"
        return None

    def reset(self):
        self.cur, self.k_now = [], None  # the reference stays: it is the validated interval's, not this one's


@dataclass
class HoldError(Clause):
    """CFD: the held correction against the fresh one, over the full time derivative (e*)."""
    e_star: float = 1.0
    name: str = "hold-error"
    kind: str = "rollback"

    def check(self, iv):
        e = iv.extra.get("hold_error")
        return None if e is None or e <= self.e_star else f"hold error {e:.3f} > {self.e_star}"


@dataclass
class RegimeEnvelope(Clause):
    """Online regime monitor: run statistics stay inside the envelope recorded when the schedule
    was validated (e.g. magnetic over kinetic energy, band enstrophy). DNS-free at run time."""
    bounds: dict = field(default_factory=dict)  # key -> (lo, hi)
    window: int = 25
    name: str = "regime"
    kind: str = "refuse"
    _hist: dict = field(default_factory=dict)

    def check(self, iv):
        for key, (lo, hi) in self.bounds.items():
            if key not in iv.extra:
                continue
            h = self._hist.setdefault(key, [])
            h.append(float(iv.extra[key]))
            if len(h) >= self.window:
                m = sum(h[-self.window:]) / self.window
                if not lo <= m <= hi:
                    return f"{key} {m:.4g} outside [{lo:.4g}, {hi:.4g}]"
        return None

    def reset(self):
        self._hist.clear()


@dataclass
class QoSThreshold(Clause):
    """Green-style quality of service: back off when the sampled output error exceeds a threshold."""
    threshold: float = 1.0
    name: str = "green-qos"
    kind: str = "refuse"

    def check(self, iv):
        return None if iv.staleness <= self.threshold else f"QoS {iv.staleness:.3f} > {self.threshold}"


@dataclass
class WorstCase(Clause):
    """Rumba-style: re-execute when the worst single output error exceeds a threshold."""
    threshold: float = 1.0
    name: str = "rumba-worst"
    kind: str = "rollback"

    def check(self, iv):
        w = iv.extra.get("worst", iv.staleness)
        return None if w <= self.threshold else f"worst {w:.3f} > {self.threshold}"
