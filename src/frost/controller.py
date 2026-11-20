"""Choosing the refresh interval online, and deciding when not to bother.

Each teacher call closes an outer step of k inner steps. It reveals, for free,
how far the residual moved over that step: e = RMS |R(x_{n+1}) - R(x_n)|. For a
linearly drifting residual the mean staleness over the step is e / sqrt(3).

The controller responds in three ways (note section 5.2, "never lose"):
* it halves k when that mean staleness exceeds the budget;
* it doubles k after a run of quiet outer steps;
* it reverts to the plain teacher when its cost model predicts less than a 5%
  gain.

Staleness and budget share whatever unit the caller measures in. Since week 3 that
unit is acceleration, meV/(A amu): integration error follows acceleration, not
force, and the same force error on a 1 amu hydrogen is 16-28x the acceleration
error it causes on oxygen or silicon (plan section 1.11). Pass a force staleness
with ``epsilon=20.0`` to reproduce the week-1 and week-2 runs.

A magnitude budget alone does not certify a multirate step, and the fidelity runs
say so (plan section 1.12). Two failure modes are distinct:

* **Magnitude.** Water at k = 2 holds a residual that is wrong by 10.8 in these
  units and heats at 0.84 meV/atom/ps.
* **Resonance.** Silica at k = 16 measures 1.11, *lower* than Li3PO4 at k = 8
  (1.23, which keeps the teacher's physics), yet it drifts 53x faster, because a
  16 fs outer step is past half the 30 fs Si-O period. This is the r-RESPA
  resonance limit, and no staleness magnitude sees it coming.

So the controller bounds both: the realised staleness against ``epsilon``, and the
outer step against ``resonance_fraction`` of the system's fastest period. The
defaults are set by measurement, not taste. ``epsilon = 2.0`` sits above the
largest staleness that kept the physics (Li3PO4's 1.92 peak) and five times below
the smallest that broke it on magnitude (water's 10.8). The cap takes ``0.4`` of
the fastest period and rounds down to an even number of steps, the lattice the
controller moves on: silica is allowed 6 inner steps of its ~30 fs period and
never 16, Li3PO4 8 of 33, water 2 of 9. That is exactly the set of outer steps
measured to keep the teacher's physics, with no extrapolation beyond them. Four
fidelity runs calibrate both numbers, so treat them as provisional until the G3
sweep is complete.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field



@dataclass
class DriftMonitor:
    """The certificate's posterior check: the energy drift the run is realising.

    The resonance cap is a prior. It predicts, from one number per system, whether
    an interval will pump the fast modes, and the measured boundary is narrow and
    not even ordered: silica passes 50 ps NVE at 0.217 of its fastest period while
    the glass is marginal at 0.211 (plan section 1.14). No fraction separates those.

    The multirate schedule computes the teacher's energy at every outer step
    anyway, so the drift itself can be watched rather than predicted. This fits a
    least-squares slope over a trailing window and reports whether it exceeds the
    budget. It is deliberately slow: a window shorter than a few picoseconds
    measures fluctuation, not drift.

    Calibrated by replaying every 50 ps run whose fidelity verdict is known --
    thirteen so far, nine passes across five systems and four failures. The window
    the evidence allows has narrowed twice as systems were added, and each time in
    the direction of the newest system:

        oxides only          admit above 0.0199, refuse below 0.0591   (3.0x wide)
        + Si                 admit above 0.0248, refuse below 0.0591   (2.4x)
        + Cu at k = 32       admit above 0.0248, refuse below 0.0348   (1.4x)

    0.029 is the geometric mean of the current window. An 0.02 fitted to the oxides
    falsely refused Si at k = 10 (0.0248); an 0.035 fitted before Cu k = 32 was
    measured falsely admitted it (0.0348), by 0.0002. The margin is now 1.4x and
    shrinking, so this is a fitted threshold and should be read as one -- the next
    system may close it entirely, which would say the statistic itself is
    insufficient rather than that the number is wrong.

    What it does not do is resolve a marginal case: the glass at k = 6 reads 0.0185
    and 0.0213 on two seeds, below Si's passing 0.0248, yet keeps gaining energy
    out to 100 ps. Transient slope is not long-run accumulation, and no threshold
    on this statistic separates those two. The monitor catches clear failures
    quickly -- 7 ps for silica and Si, 10 ps for Li3PO4 -- and leaves marginal
    intervals to the controller's own fallback.
    """

    budget: float = 0.029  # meV per atom per ps, for the fitted slope test
    window_ps: float = 10.0
    horizon_ps: float = 50.0  # the length G3 judges a run over
    gain_budget: float = 1.0  # meV per atom by the horizon: G3's criterion itself
    min_ps: float = 20.0  # shorter than this, an early transient reads as drift
    history: list = field(default_factory=list)
    adopted: list = field(default_factory=list)
    # Sequential mode (opt-in, md_runtime --drift-sequential; gaps.md E.9). The fixed rule above decides only once
    # min_ps of an interval exists, and reset() forgets what earlier intervals spent, so Si at 1000 K ran 20 ps at
    # k 10 and another 20 ps at k 8 before either refusal and gained +1.68 meV/atom over 50 ps against a budget of 1.0
    # (refusals at exactly 20.0 and 40.0 ps). Two changes, both read off G3's own definition:
    #   1. group-sequential looks (O'Brien and Fleming 1979): once eight 1 ps block means exist (7 ps), refuse when
    #      the excess clears z x sqrt(min_ps / span) standard errors -- 3.4 at 7 ps, 2.83 at 10 ps, 2.0 (the fixed
    #      rule) from 20 ps -- on 1 ps block means, the statistic validate.py records and the replay was run on;
    #   2. a run ledger: G3 bounds the gain over any 50 ps window, not per interval, so the slope an interval may
    #      have is what the window that starts before it has left: (budget - spent) / (horizon - elapsed), floored
    #      at zero. reset() keeps the ledger (only restart() clears it).
    # Replayed on the 75 traces of 50 ps NVE runs on disk (19 energy failures): the mean time to refuse a failure
    # falls from 21.6 ps to 11.2 ps; false admits stay 1 (silica k 7, gain 1.033, admitted by the fixed rule too);
    # passes refused rise from 1 to 3: Cu k 32 twice (gains 0.78; it fails G3 on structure) and MgO k 10 (0.93).
    sequential: bool = False
    early_ps: float = 5.0
    block_ps: float = 1.0
    z: float = 2.0
    # Version 3: a CUSUM (Page 1954) on the record's blocks against the allowed rate, for an energy that rises late after a
    # flat stretch, which one line fitted to the whole record dilutes: Li3PO4 at 1000 K (predictive arm) sat near 0.42
    # meV/atom from 8 to 20 ps, then climbed to 1.63 by 50 ps with no refusal. It fires when the energy has risen above the
    # allowed line, measured from that line's lowest point, by cusum_z robust standard deviations of a block-to-block change
    # (1.4826 x MAD) and by at least cusum_floor of the budget. Replay with the slope test on the 75 known traces: 0 false
    # admits (the slope test alone has 1), 5 passes refused (Cu k 32 three times, failing on structure; MgO k 10 at 0.91,
    # 0.93); on that Li3PO4 trace it fires at 24 ps (0.58), and it stays silent on every version 2 run that passed.
    cusum_z: float = 8.0
    cusum_floor: float = 0.05
    last_rates: tuple | None = None  # (measured, allowed) meV/atom/ps at the last refusal (step_factor)
    blocks: list = field(default_factory=list)  # (t, mean energy) of the current interval's closed blocks
    ledger: list = field(default_factory=list)  # the run's closed blocks over the trailing horizon; reset() keeps it
    _open: list = field(default_factory=list)  # the block being filled

    @staticmethod
    def _append(points: list, t_ps: float, energy: float) -> None:
        """Record a sample, replacing rather than duplicating one at the same time.

        A caller that reads the projection and then asks whether it is over budget
        would otherwise feed the same sample twice and bias the fit toward whatever
        it read last. Making a repeated timestamp idempotent means any order of
        ``projected_gain``, ``over_projection``, ``observe`` and ``over_budget``
        within one outer step gives the same answer.
        """
        if points and points[-1][0] == t_ps:
            points[-1] = (t_ps, energy)
            return
        points.append((t_ps, energy))

    @staticmethod
    def _fit(points: list) -> tuple[float, float] | None:
        """Least-squares slope and its standard error, from the residuals."""
        if len(points) < 8:
            return None
        t = [point[0] for point in points]
        e = [point[1] for point in points]
        n = len(t)
        mean_t = sum(t) / n
        mean_e = sum(e) / n
        variance = sum((a - mean_t) ** 2 for a in t)
        if variance <= 0.0:
            return None
        slope = sum((a - mean_t) * (b - mean_e) for a, b in zip(t, e, strict=True)) / variance
        intercept = mean_e - slope * mean_t
        residual = sum((b - (intercept + slope * a)) ** 2 for a, b in zip(t, e, strict=True))
        return slope, math.sqrt(residual / ((n - 2) * variance)) if n > 2 else float("inf")

    @staticmethod
    def _slope(points: list) -> float | None:
        fit = DriftMonitor._fit(points)
        return None if fit is None else fit[0]

    def reset(self) -> None:
        """Start again: the interval changed, so what came before describes another one.

        The ledger stays: energy an earlier interval gained is still in the system and still inside the window G3
        judges. The open block is closed into the ledger only (it straddles the change, so it is no interval's).
        """
        self.history.clear()
        self.adopted.clear()
        if self._open:
            self._close(into_interval=False)
        self.blocks.clear()

    def truncate(self, t_ps: float) -> None:
        """The executor rewound the trajectory to ``t_ps``: forget every sample from then on, in every record."""
        for points in (self.history, self.adopted, self.blocks, self.ledger, self._open):
            points[:] = [q for q in points if q[0] < t_ps]

    def restart(self) -> None:
        """A new run: forget everything, the ledger included."""
        self.reset()
        self.ledger.clear()
        self._open.clear()

    def _close(self, into_interval: bool = True) -> None:
        t = sum(p[0] for p in self._open) / len(self._open)
        e = sum(p[1] for p in self._open) / len(self._open)
        if into_interval:
            self.blocks.append((t, e))
        self.ledger.append((t, e))
        while self.ledger and self.ledger[-1][0] - self.ledger[0][0] > self.horizon_ps:
            self.ledger.pop(0)
        self._open.clear()

    def _add_block_point(self, t_ps: float, energy: float) -> None:
        if self._open and self._open[-1][0] == t_ps:  # idempotent on a repeated time (_append)
            self._open[-1] = (t_ps, energy)
            return
        if self._open and t_ps - self._open[0][0] >= self.block_ps:
            self._close()
        self._open.append((t_ps, energy))

    def allowed_rate(self, blocks: list, slope: float) -> float:
        """The slope (meV/atom/ps) the record ``blocks`` may have: the budget's rate, or less when the 50 ps window that
        starts before the record has already spent part of it (the ledger)."""
        allowed = self.gain_budget / self.horizon_ps  # the window that starts with this record
        t_a = blocks[0][0]
        past = [b for b in self.ledger if b[0] < t_a - 0.5 * self.block_ps and b[0] >= t_a - self.horizon_ps]
        if len(past) >= 2:
            # the window that starts with the oldest past block inside the horizon: what it has spent by t_a (the
            # record's fitted level there, less the mean of the two oldest blocks), and the time it has left
            t_s = 0.5 * (past[0][0] + past[1][0])
            level = sum(b[1] for b in blocks) / len(blocks) + slope * (t_a - sum(b[0] for b in blocks) / len(blocks))
            spent = level - 0.5 * (past[0][1] + past[1][1])
            left = self.horizon_ps - (t_a - t_s)
            if left > 0.0:
                allowed = min(allowed, max(self.gain_budget - spent, 0.0) / left)
        return allowed

    def sequential_refuse(self, energy_meV_per_atom: float, t_fs: float) -> bool:
        """Group-sequential test of the interval's slope against the rate its 50 ps windows allow (see the fields)."""
        self._add_block_point(t_fs / 1e3, energy_meV_per_atom)
        span = self.blocks[-1][0] - self.blocks[0][0] if self.blocks else 0.0
        if span < self.early_ps:
            return False
        fit = self._fit(self.blocks)  # eight blocks at least, as in the replay: the first look comes at 7 ps
        if fit is not None:
            slope, se = fit
            z = self.z * math.sqrt(max(self.min_ps / span, 1.0))
            allowed = self.allowed_rate(self.blocks, slope)
            if slope - allowed > z * se:
                self.last_rates = (slope, allowed)
                return True
        if self._cusum(fit[0] if fit is not None else 0.0):
            # the record's mean rate (the CUSUM can decide before a fit exists)
            (t0, e0), (t1, e1) = self.blocks[0], self.blocks[-1]
            rate = (e1 - e0) / max(t1 - t0, 1e-9)
            self.last_rates = (rate, self.allowed_rate(self.blocks, fit[0] if fit is not None else rate))
            return True
        return False

    def step_factor(self, p: float = 1.5) -> float | None:
        """After a refusal: the fraction of the refused interval expected to bring the measured rate under the allowed one,
        (allowed / measured)^(1/p) with drift ~ k^p (Si at 1000 K: p 1.2-1.8 between k 6, 8, 10), or None if unknown."""
        if self.last_rates is None:
            return None
        measured, allowed = self.last_rates
        if measured <= 0.0:
            return None
        return (max(allowed, 0.0) / measured) ** (1.0 / p)

    def _cusum(self, slope: float) -> bool:
        if len(self.blocks) < 5:
            return False
        rate = self.allowed_rate(self.blocks, slope)
        t0 = self.blocks[0][0]
        line = [e - rate * (t - t0) for t, e in self.blocks]
        excess = line[-1] - min(line)
        steps = [b[1] - a[1] for a, b in zip(self.blocks, self.blocks[1:], strict=False)]
        med = statistics.median(steps)
        mad = 1.4826 * statistics.median(abs(x - med) for x in steps)
        return excess > max(self.cusum_z * mad, self.cusum_floor * self.gain_budget)

    def admits(self, since_ps: float) -> bool:
        """Probation (sequential mode): the record since ``since_ps`` (one interval's) has eight blocks and its slope,
        plus two standard errors, sits under half the rate the ledger allows -- ThermostatBias's admission margin, which
        leaves room for the next rung's larger drift (Si at 1000 K: 0.026 at k 6, 0.037 at k 8, 0.055 at k 10)."""
        blocks = [b for b in self.blocks if b[0] >= since_ps]
        fit = self._fit(blocks) if len(blocks) >= 8 else None
        if fit is None:
            return False
        return fit[0] + 2.0 * fit[1] < 0.5 * self.allowed_rate(blocks, fit[0])

    def projected_gain(self, energy_meV_per_atom: float, t_fs: float) -> float | None:
        """Energy this interval would accumulate by the horizon, in meV/atom.

        The slope test above needs a budget fitted to known-good runs, and that
        budget has been wrong twice: the window of admissible values narrowed from
        3.0x to 1.4x as systems were added, each time in the direction of the
        newest one. This instead extrapolates what has accumulated since the
        interval was adopted out to ``horizon_ps`` and compares it with
        ``gain_budget`` -- which is not a fitted number at all but the criterion
        G3 already judges runs by, 1 meV/atom over 50 ps.

        Replayed over the same thirteen runs, projecting from 20 ps: the worst
        passing run reaches 0.53 and the worst energy-failure 1.87, a 3.5x window
        around a threshold nothing was tuned to. From 10 ps it is 1.0x and useless
        -- Li3PO4 at k = 6 projects to 0.85 there and 0.11 by 20 ps, an early
        transient that decays -- which is why ``min_ps`` is 20 and why this clause
        is slower than the slope test it supersedes (frost-87's proposal).

        Cu at k = 32 projects to 0.39 and is admitted here: it fails on structure,
        not energy, and is the resonance clause's case rather than this one's.
        """
        self._append(self.adopted, t_fs / 1e3, energy_meV_per_atom)
        span = self.adopted[-1][0] - self.adopted[0][0]
        if span < self.min_ps:
            return None
        slope = self._slope(self.adopted)
        return None if slope is None else slope * self.horizon_ps

    def over_projection(self, energy_meV_per_atom: float, t_fs: float) -> bool:
        """Refuse only when the projection exceeds the budget *significantly*.

        The projection is a fitted slope and carries the fit's own uncertainty, so
        comparing it with the budget directly refuses on noise. Li3PO4 at k = 6 was
        refused online at a projection of 1.0044 against a budget of 1.0 -- four
        parts in a thousand -- on an interval that fixed-k fidelity passes
        comfortably (+0.07 meV/atom over 50 ps, flat halves), and it cost that
        system 3.67x -> 3.00x (frost-ec). Requiring the excess to clear two
        standard errors costs no fitted constant and separates every run we have:
        on the thirteen traces of known verdict the worst pass sits 5.7 standard
        errors *below* the budget and the worst energy-failure 10.0 above it.

        With ``sequential`` set, the decision is ``sequential_refuse`` instead.
        """
        if self.sequential:
            return self.sequential_refuse(energy_meV_per_atom, t_fs)
        gain = self.projected_gain(energy_meV_per_atom, t_fs)
        if gain is None:
            return False
        fit = self._fit(self.adopted)
        if fit is None:
            return False
        error = fit[1] * self.horizon_ps
        return gain - self.gain_budget > 2.0 * error

    def observe(self, energy_meV_per_atom: float, t_fs: float) -> float | None:
        """Record a point; return the trailing drift in meV/atom/ps once measurable."""
        self._append(self.history, t_fs / 1e3, energy_meV_per_atom)
        span = self.history[-1][0] - self.history[0][0]
        while span > self.window_ps and len(self.history) > 2:
            self.history.pop(0)
            span = self.history[-1][0] - self.history[0][0]
        if span < 0.5 * self.window_ps:
            return None
        return self._slope(self.history)

    def over_budget(self, energy_meV_per_atom: float, t_fs: float) -> bool:
        drift = self.observe(energy_meV_per_atom, t_fs)
        return drift is not None and abs(drift) > self.budget


@dataclass
class IntervalController:
    epsilon: float = 2.0
    k: int = 1
    k_max: int = 16
    period_fs: float | None = None  # fastest vibrational period; None disables the cap
    dt_fs: float = 1.0
    resonance_fraction: float = 0.22
    k_jitter: float = 0.0  # draw each outer step from k * [1 - jitter, 1 + jitter]
    quiet_to_grow: int = 5
    alpha: float = 0.3
    c_student: float = 0.05
    o_verify: float = 0.02
    o_control: float = 0.01
    min_gain: float = 1.05
    ewma: float | None = None
    quiet: int = 0
    ceiling: int | None = None  # ratcheted down by refuse(); never rises again
    predictive: bool = False  # error-per-step control (observe_predictive) instead of halve / double
    safety: float = 0.8  # predictive: aim at this fraction of epsilon
    # optional zero-argument callable multiplying epsilon wherever the controller reads it, the same one the magnitude
    # clause uses (md_runtime --thermal-epsilon): one bound, not two copies that can disagree (gaps.md E.7 L6)
    epsilon_scale: object = None
    # the single-interval shrink test's allowance for per-interval scatter, the same number the magnitude clause's
    # rollback threshold carries (Magnitude.allowance, md_runtime --small-n-allowance; 1.0 = none). One bound, not two
    # copies: on one ethanol (9 atoms) the clause allowed 5.8 while this test halved k at 4.03, so single noisy intervals
    # (max 8.0, mean 3.4) dropped the controller to k 1, which never-lose reads as "the cheap path does not pay"
    # (gaps.md, DFT teacher, 2026-11-15). Growth still compares the running mean with epsilon itself.
    interval_allowance: float = 1.0
    # with an allowance (> 1: a few-atom system), the controller judges the running mean with its own error bar, not
    # single intervals: it steps down when the mean is above epsilon by growth_z of the mean's standard error (the EWMA's
    # sd, sqrt(var x alpha / (2 - alpha)) from an EW variance of the per-interval staleness), and grows only when the mean
    # plus that margin would still fit at the next rung. Without these the shared bound let one ethanol wander to k 3 on
    # dips of a noisy 3-interval average, where its mean staleness (5.2) sat above epsilon (4.03) but under the allowance,
    # and 277 intervals rolled back (gaps.md, DFT teacher, third run); a mean rule without the margin stepped k 2 (mean 3.4)
    # down on the average's own noise. Both are off at interval_allowance 1.0 (every other system).
    growth_z: float = 2.0
    ewvar: float = 0.0
    n_obs: int = 0
    # the small-N statistics: a slow EWMA (memory ~1 / small_alpha intervals) of the per-interval staleness and its EW
    # variance, rescaled when k changes (staleness ~ k, as the growth rule assumes), fed by accepted AND rolled-back
    # intervals (observe_rollback: a rollback is a measurement at that k too; leaving it out censors the mean low)
    small_alpha: float = 0.05
    ewma_s: float | None = None
    ewvar_s: float = 0.0
    n_at_k: int = 0
    k_seen: int = 0
    growth: float = 1.0  # predictive: staleness ~ k^growth, refitted online
    last: tuple | None = None  # predictive: (k, staleness) of the previous accepted interval
    history: list = field(default_factory=list)
    refusals: list = field(default_factory=list)
    # Probationary growth (NVT measured clauses): growth may not pass ``probation`` until every admitter has judged
    # the current interval and admitted it; each full admission raises it one rung (x1.25, on the lattice). None
    # disables it. Without it the halve / double rule doubled MgO past the validated prior to k 16 and 22 before the
    # bias clause could judge either (5 ps), and each refusal stepped down one rung per 5 ps: +3.7 K over 20 ps
    # against a 1.5 K tolerance.
    probation: int | None = None
    # Growth test scaled to the step taken (opt-in, used with probation): grow to k_next only if the staleness
    # predicted there, ewma x k_next / k (linear in k), stays under growth_margin x epsilon. The default rule,
    # ewma < epsilon / 2, is that test for a doubling; probation steps one rung (x1.33 at k 6), where it refused
    # PET silica's k 6 -> 8 at staleness 1.19 although k 8 runs at 1.52 mean / 1.96 max under epsilon 2.0.
    linear_growth: bool = False
    growth_margin: float = 0.9
    admitters: tuple = ()
    admitted: dict = field(default_factory=dict)  # k -> names that admitted it
    # (k, name) -> how far that admitter's measured margin lets the next rung go (ThermostatBias with predictive
    # admission, 2026-11-14); admitters that give none count as the default 1.25
    admit_factors: dict = field(default_factory=dict)
    # NVE, md_runtime --drift-sequential (gaps.md E.9): after a measured refusal growth becomes probationary (above).
    # Without it Si at 1000 K, refused at k 10, climbed straight back to k 8 within tens of fs and spent another 14 ps
    # there before the drift clause could refuse it, then the same at k 6: +1.18 meV/atom over 50 ps.
    probation_on_refusal: bool = False
    # The fine lattice (opt-in, md_runtime --fine-lattice FS): below FS femtoseconds every step count is a lattice point.
    # The even lattice exists so that silica's validated k 6 is reachable; at the small end it only removes intervals:
    # water's prior (0.22 of 16.8 fs = 3.7 fs) and Li3PO4's halving (6 -> 3) both landed on k 2, while fixed k 3 runs
    # water at 1.85x against k 2's 1.38x and held Li3PO4's energy at 1000 K (gaps.md E.8, E.9).
    fine_fs: float = 0.0

    def _bound(self) -> int:
        bound = min(self.k_max, self.resonance_cap(), self.ceiling or self.k_max)
        return bound if self.probation is None else min(bound, self.probation)

    def start_probation(self, prior_k: int, floor: int = 2) -> int:
        """Growth past ``prior_k`` (the NVE-validated prior) becomes probationary, but never below ``floor``, the
        smallest interval that pays (the executor's probe interval): below it the never-lose rule runs the baseline,
        whose intervals no measured clause sees, so a prior under two steps could never be admitted past. The peptide
        without heavy hydrogens (0.22 x 8.40 fs = 1.85 fs at 1 fs, a prior of one step) cycled for 50 ps between a brief
        k 2 after each never-lose resume and the baseline: 49,790 of 50,000 steps on the teacher alone (2026-11-14)."""
        self.probation = max(int(prior_k), int(floor))
        return self.probation

    def admit(self, k: int, source: str, factor: float | None = None) -> None:
        """A measured clause judged interval ``k`` and found it inside its tolerance with margin.

        ``factor`` (optional) is how far that clause's measured margin says the next rung may go; the rung taken is the
        smallest any admitter allows, 1.25 for an admitter that gives none, so a clause without a margin estimate (the
        structure clause) keeps the one-rung climb."""
        if self.probation is None or k < self.probation:
            return
        names = self.admitted.setdefault(k, set())
        names.add(source)
        if factor is not None:
            self.admit_factors[(k, source)] = float(factor)
        if set(self.admitters) <= names:
            g = min(self.admit_factors.get((k, s), 1.25) for s in self.admitters)
            self.probation = max(self.probation + self.lattice_step(), self.even(int(math.ceil(g * k))))

    def resonance_cap(self) -> int:
        """Largest outer step the fastest mode tolerates, in inner steps.

        Calibrated on 50 ps NVE runs, which is the length that settles it: a 10 ps
        trace can look flat while the energy is still climbing. Against periods
        measured from the teacher's own spectrum (`observables.fastest_period_fs`),
        the runs that hold the teacher's energy sit at 0.182 and 0.217 of the
        fastest period and the ones that fail at 0.242 and 0.289, so the boundary
        is between 0.217 and 0.242 — far below the 0.4 that half-period linear
        stability suggests. The fraction is 0.22, just above the highest passing
        run: 0.2 would refuse silica at k = 6, which two 50 ps seeds say is fine,
        and cost it a quarter of its speedup.

        The fraction alone cannot be trusted to separate systems, and is not asked
        to: silica passes at 0.217 while the glass is marginal at 0.211, an
        inverted ordering of the same kind that refuted a single staleness budget
        in section 1.12. ``DriftMonitor`` is the posterior check that catches what
        this prior admits wrongly.

        The lattice is even integers, not doublings. Silica's passing k = 6 is not
        a power of two, so the old lattice could not express it and handed out 8,
        which fails at 50 ps (+1.81 meV/atom) while passing every 10 ps check.
        """
        if not self.period_fs:
            return self.k_max
        # The lattice is even in femtoseconds, then converted to steps: the clause bounds
        # the interval as a time. Rounding the step count instead refused silica at dt = 2,
        # H = 6 fs (k = 3, which passes 50 ps twice: +0.357 and +0.526 meV/atom), because
        # 3 is odd (qos_headtohead.py found it).
        limit_fs = self.resonance_fraction * self.period_fs
        if limit_fs <= self.fine_fs:  # the fine lattice: every step count below fine_fs (water's 3.7 fs gives k 3)
            return max(int(limit_fs / self.dt_fs + 1e-9), 1)
        even_fs = int(limit_fs) - (int(limit_fs) % 2)
        steps = int(even_fs / self.dt_fs + 1e-9)
        return max(steps, 1)

    def even(self, k: int) -> int:
        """The interval lattice: intervals even in femtoseconds, so a passing k = 6 is reachable.

        The lattice bounds a time, as ``resonance_cap`` does: at dt = 1 fs it is even step counts, at
        dt = 2 fs every step count (k = 5 is 10 fs). Rounding the step count instead held Si at k = 4 under
        a k_max of 5 at dt = 2 fs (md_si512_T300_fair_dt2.json: 3.09x against 3.69x at the 10 fs interval).
        """
        if k <= 2 or k * self.dt_fs <= self.fine_fs:
            return k
        step = self.lattice_step()
        return max(k - (k % step), 1)

    def rung_below(self, k: int) -> int:
        """The lattice point just below k: one step on the fine lattice, otherwise the even lattice's next point down."""
        if (k - 1) * self.dt_fs <= self.fine_fs:
            return max(1, k - 1)
        return self.even(max(1, k - 2))

    def lattice_step(self) -> int:
        """Smallest step count whose duration is an even number of femtoseconds (2 for a non-integer dt)."""
        for m in (1, 2, 4, 8):
            fs = m * self.dt_fs
            if abs(fs - round(fs)) < 1e-9 and round(fs) % 2 == 0:
                return m
        return 2

    def draw(self, gen, k: int | None = None) -> int:
        """The outer step to run next, jittered around ``k``.

        r-RESPA's resonance is coherent: energy is pumped when the outer step keeps
        a fixed ratio to a vibration, which is why silica drifts at k = 8 (near a
        third of its ~30 fs period) while conserving energy at k = 4. Drawing the
        interval independently of the state breaks that coherence, exactly as a
        random trajectory length does in HMC (Mackenzie 1989), and costs nothing:
        the mean interval, and so the speedup, is unchanged.
        """
        k = self.k if k is None else k
        if self.k_jitter <= 0.0 or k < 2:
            return k
        # the window is symmetric about k, so the mean interval — and the speedup —
        # is what the controller chose. The resonance cap bounds that mean, not each
        # draw: a cap equal to k would otherwise force every draw downwards, and the
        # magnitude budget still judges each outer step as it closes.
        spread = max(1, round(k * self.k_jitter))
        lo = max(1, k - spread)
        import torch  # only the jittered draw needs it; the JAX environments import this module too

        return int(torch.randint(lo, k + spread + 1, (), generator=gen))

    def predicted_speedup(self, k: int | None = None) -> float:
        """Eq. (8) of the note, in teacher force calls; k = 1 is the plain teacher."""
        k = self.k if k is None else k
        if k <= 1:
            return 1.0 / (1.0 + self.o_control)
        return 1.0 / (self.c_student + (1.0 + self.o_verify) / k + self.o_control)

    def refuse(self, k_used: int, factor: float | None = None) -> int:
        """A measured clause rejected ``k_used``: drop now, and never re-adopt it.

        Growth and refusal run on wildly different clocks. The magnitude clause
        grows after ``quiet_to_grow`` quiet outer steps — 480 fs at k = 16 — while
        a measured clause needs ``DriftMonitor.min_ps``, 20,000 fs, to judge the
        interval at all. So a refusal that does not latch is undone before the
        monitor can restate it: on Si the projection clause refused k = 16 at
        20.3 ps and the growth rule was back at 16 forty femtoseconds later, so the
        run spent 40 of its 60 ps at an interval already measured to fail. Any
        measured clause is outvoted by the growth rule unless its refusals stick.

        Two effects, and the second is what makes the first safe to keep simple:

        - the interval halves at once, which is the fast, safe move;
        - the ceiling ratchets to one rung *below* the refused interval, so the
          growth rule can climb back toward the boundary without ever re-adopting
          a measured failure. On Si that converges 16 → 8, ceiling 14 → refused
          → ceiling 12 → refused → ceiling 10, which is the interval Si actually
          passes at (4.22x) and one rung better than halving alone would reach
          (k = 8, 3.77x). Each rung costs another ``min_ps`` of evidence, so this
          converges over a long run and not inside a 60 ps one.
        """
        floor = self.rung_below(k_used)
        self.ceiling = floor if self.ceiling is None else min(self.ceiling, floor)
        k_new = k_used // 2
        if factor is not None:
            # measured step-down (md_runtime --drift-step, only with --drift-rewind, which undoes any overshoot): drop by
            # the measured excess, between halving and one rung; AgI's energy wandered 0.03 against 0.02 meV/atom/ps at
            # k 36 and halving to 18 cost 14-27% against fixed 36 (gaps.md E.8)
            k_new = max(k_used // 2, min(int(k_used * factor), k_used - self.lattice_step()))
        self.k = self.even(max(1, min(k_new, self.ceiling)))
        self.quiet = 0
        if self.probation_on_refusal:
            # a measured refusal says the prior is too generous here: from now on every rung past the new interval
            # must be admitted by the measured clauses first (admit()), and earlier admissions describe another state
            self.probation = self.k if self.probation is None else min(self.probation, self.k)
            self.admitted.clear()
            self.admit_factors.clear()
        self.refusals.append((k_used, self.k, self.ceiling))
        return self.k

    def eps(self) -> float:
        """The bound in force now: epsilon, times epsilon_scale() when one is set."""
        return self.epsilon * (float(self.epsilon_scale()) if self.epsilon_scale is not None else 1.0)

    def _small_stats(self, h: float, k_used: int) -> None:
        """Update the small-N statistics with one interval's staleness h at k_used (k 1, the remnant of a split
        interval, is left out)."""
        if k_used <= 1:
            return
        if self.ewma_s is None:
            self.ewma_s, self.ewvar_s, self.k_seen, self.n_at_k = h, 0.0, k_used, 1
            return
        if k_used != self.k_seen:
            r = k_used / max(self.k_seen, 1)
            self.ewma_s *= r
            self.ewvar_s *= r * r
            self.k_seen, self.n_at_k = k_used, 0
        d = h - self.ewma_s
        a = self.small_alpha
        self.ewma_s += a * d
        self.ewvar_s = (1.0 - a) * (self.ewvar_s + a * d * d)
        self.n_at_k += 1

    def reset_small_stats(self) -> None:
        """Forget the small-N statistics: the executor calls this when never-lose resumes on clean probes, which are fresh
        evidence. Without it the statistics, frozen through the baseline stretch, kept the value a transient hot stretch
        had left (one ethanol, fourth run: a stretch of rollbacks at 0.4-0.6 ps raised the slow mean past epsilon; after
        each of six resumes the first accepted interval stepped k 2 down to 1 again, and 7,390 of 8,000 steps ran on the
        teacher)."""
        self.ewma_s, self.ewvar_s, self.n_at_k, self.k_seen = None, 0.0, 0, 0

    def observe_rollback(self, staleness: float, k_used: int) -> int:
        """A rolled-back interval (small-N mode only; otherwise nothing changes): its staleness enters the statistics, and
        above the never-lose rung (k 2) the interval steps down at once, so a failing rung is not retried on the next
        interval. At k 2 and below the never-lose cost test owns the decision."""
        if self.interval_allowance <= 1.0:
            return self.k
        self._small_stats(staleness / math.sqrt(3.0), k_used)
        if k_used > 2:
            self.k = self.even(max(2, k_used // 2))
            self.quiet = 0
        return self.k

    def observe(self, staleness: float, k_used: int) -> int:
        """Record the realised staleness of the outer step that just closed; return the next k."""
        if self.predictive:
            return self.observe_predictive(staleness, k_used)
        h = staleness / math.sqrt(3.0)
        if self.ewma is None:
            self.ewma = h
        else:
            d = h - self.ewma
            self.ewma = self.ewma + self.alpha * d
            self.ewvar = (1.0 - self.alpha) * (self.ewvar + self.alpha * d * d)
        self.n_obs += 1
        small_n = self.interval_allowance > 1.0
        if small_n:
            self._small_stats(h, k_used)
            a = self.small_alpha
            mean = self.ewma_s if self.ewma_s is not None else self.ewma
            sd_mean = math.sqrt(self.ewvar_s * a / (2.0 - a))
        else:
            mean, sd_mean = self.ewma, 0.0
        level = mean + self.growth_z * sd_mean
        k_next = self.even(min(self._bound(), max(2, 2 * k_used)))
        if self.linear_growth:
            room = level * k_next / max(k_used, 1) < self.growth_margin * self.eps()
        else:
            # the doubling rule, ewma < epsilon / 2, written for the step actually taken (2026-11-14): k_next is a
            # doubling unless a cap or the fine lattice shortens it, and then the halving of epsilon refused a step
            # that is safe. The peptide in NVE sat at k 2 for 50 ps (staleness 3.76 against epsilon / 2 = 3.16) while
            # its cap and the fine lattice allowed 3, which runs at 5.2 under epsilon 6.32 (fixed 3: 1.70x against
            # 1.13x, both passing). Identical to the old rule for a doubling (k_next = 2 k_used).
            room = level * k_next / max(k_used, 1) < self.eps()
        if small_n and self.n_at_k < 10:
            room = False  # grow on a measured mean at this rung, not on its first few noisy intervals
        mean_over = small_n and self.n_at_k >= 5 and mean - self.growth_z * sd_mean > self.eps()
        if h > self.eps() * self.interval_allowance or mean_over:
            # halve, but from 3 (the fine lattice) to 2, not to 1: k 1 is dearer than the baseline, so never-lose would
            # read a step down from 3 as "the cheap path does not pay"
            self.k = self.even(max(1 if k_used <= 2 else 2, k_used // 2))
            self.quiet = 0
        elif room:
            self.quiet += 1
            if self.quiet >= self.quiet_to_grow:
                self.k = k_next
                self.quiet = 0
        else:
            self.quiet = 0
        self.history.append((k_used, staleness, self.k))
        return self.k

    def observe_predictive(self, staleness: float, k_used: int) -> int:
        """Error-per-step control: choose the next k from the measured staleness and how it grows with k.

        The halve / double rule overshoots whenever conditions drift. On Cu heated to 3500 K it rolled back 5,670 of
        about 8,300 intervals: each doubling after five quiet intervals crossed epsilon, the executor redid the
        interval in halves, and the cycle repeated, so two thirds of the teacher calls bought nothing. Adaptive ODE
        solvers solved this decades ago (Gustafsson 1991; Soderlind 2003): model the error as e(k) = c k^p, refit p
        from consecutive intervals of different length, and pick the k whose predicted error is a safety fraction
        of the tolerance, k_new = k (safety * epsilon / e)^(1/p), limited to a factor of two either way. The clauses
        are untouched: an interval that still overshoots is rolled back as before, and caps and ceilings still bind.
        """
        h = staleness / math.sqrt(3.0)
        self.ewma = h if self.ewma is None else self.alpha * h + (1.0 - self.alpha) * self.ewma
        if self.last is not None and self.last[0] != k_used and h > 0.0 and self.last[1] > 0.0:
            p = math.log(h / self.last[1]) / math.log(k_used / self.last[0])
            self.growth = 0.7 * self.growth + 0.3 * min(3.0, max(0.5, p))
        if h > 0.0:
            self.last = (k_used, h)
        factor = 2.0 if h <= 0.0 else (self.safety * self.eps() / h) ** (1.0 / self.growth)
        k_new = int(k_used * min(2.0, max(0.5, factor)))
        self.k = self.even(max(1, min(self._bound(), k_new)))
        self.quiet = 0
        self.history.append((k_used, staleness, self.k))
        return self.k

    def use_student(self) -> bool:
        """Whether the multirate path is worth running at the current interval."""
        return self.k > 1 and self.predicted_speedup(self.k) >= self.min_gain
