"""The certified multirate executor, independent of framework and domain.

    while t < horizon:
        k = min(controller's choice, every prior's cap, ceiling)
        checkpoint (if any clause can roll back)
        interval = backend.interval(k, mode)      # k cheap steps + one refresh, per mode
        for each clause: reason = clause.check(interval)
            rollback clause -> restore, redo as two halves (down to k = 1)
            refuse clause   -> latch: halve now, ceiling one rung below for good (controller.refuse)
        controller.observe(staleness, k)

Modes are the backend's business (the executor only names them):
- "sync": the refresh at each boundary, before the next interval (MD: boundary r-RESPA);
- "lag": the refresh one interval behind, on a second device (MD: C2; NVT only);
- "spec": the refresh at a predicted boundary state, its result transported to the true one,
  and the interval integrated with the refreshed quantity ramped across it.

Never lose (note section 5.2), with ``never_lose=True`` and a backend that has ``baseline_interval``: when the
smallest multirate interval (``probe_k``) is rolled back, or the cost model predicts no gain, the executor runs the
baseline itself (MD: the teacher every step) in chunks of ``probe_every`` steps. Each chunk ends with a probe, the
cheap path evaluated at two points ``probe_k`` steps apart, judged by the rollback clauses as if it were an interval.
Multirate resumes after ``controller.quiet_to_grow`` clean probes in a row; every failed probe doubles the chunk (up
to ``probe_every_max``), so a system the certificate never admits costs the baseline plus two cheap calls per chunk.

Every decision is logged, so a run's certificate can be audited after the fact.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from frost.controller import IntervalController


@dataclass
class Interval:
    t: float  # simulated time at the interval's end (fs for MD, steps for CFD)
    k: int
    staleness: float  # the held quantity's change over the interval, in the magnitude clause's units
    finite: bool = True
    energy: float | None = None  # meV/atom, NVE only: the drift clause's input
    extra: dict = field(default_factory=dict)


class Executor:
    def __init__(self, backend, clauses, controller: IntervalController | None = None, mode: str = "sync",
                 k: int = 0, log_every: int = 0, never_lose: bool = False, probe_every: int = 100, probe_k: int = 2,
                 probe_every_max: int = 3200):
        self.backend, self.clauses, self.mode = backend, list(clauses), mode
        self.never_lose = bool(never_lose) and not k and hasattr(backend, "baseline_interval")
        self.on_baseline = False
        self.probe_every, self.probe_k, self.probe_every_max = probe_every, probe_k, probe_every_max
        self.quiet_probes = 0
        self.baseline_steps = 0
        # never lose, as a cost test (2026-11-14): the failure rate at the smallest interval, an EWMA over its attempts
        # (multirate intervals and probes). One failure used to latch the baseline: the peptide without heavy
        # hydrogens, at k 2 with 1 rollback in ~900 intervals, spent 47% of 50 ps on the teacher alone (28 latches,
        # each >= 500 baseline steps before five clean probes resumed it).
        self.small_rate, self.small_alpha = 0.0, 0.02
        self._cost_fallback = False
        self.controller = controller or IntervalController()
        self.fixed_k = k
        self.log, self.events = [], []
        self.log_every = log_every
        self.cap = None
        # every prior clause bounds k on the controller's own lattice (one source for the cap, 2026-11-14)
        caps = [c.cap({"fine_fs": getattr(self.controller, "fine_fs", 0.0)}) for c in self.clauses]
        caps = [c for c in caps if c]
        if caps:
            self.cap = min(caps)
        self.rollback = any(c.kind == "rollback" for c in self.clauses)
        # Record-level rewind (opt-in: a refusing clause with ``rewind = True``, md_runtime --drift-rewind; gaps.md E.8).
        # A measured clause needs several ps of record before it can refuse, and a resonant interval can spend the whole
        # NVE energy budget inside that time (beta-Ti at 1300 K: 2.16 meV/atom in 6 ps at its first k 26; Li, one start:
        # 0.99 in 5 ps). Refusal alone keeps that energy in the trajectory. So the executor checkpoints the state whenever
        # the controller adopts a new interval (a segment), and when a rewind clause refuses, it restores that checkpoint
        # and drops the segment's log: the trajectory never carries the refused interval. The per-interval rollback does
        # the same one level down; the wasted compute stays in the timing.
        self.rewinding = any(getattr(c, "rewind", False) for c in self.clauses)
        self.cap_bound = 0
        self.segment = None  # (checkpoint, k, log length) at the adoption of the current interval
        self.rewound = False

    def _k_now(self, remaining: int) -> int:
        k = self.fixed_k or self.controller.k
        if self.cap is not None and not self.fixed_k:
            if self.cap < k:
                self.cap_bound += 1  # the controller wanted more than a prior allows: counted, so a trace can see it
            k = min(k, self.cap)
        if self.controller.ceiling is not None:
            k = min(k, self.controller.ceiling)
        return max(1, min(k, remaining))

    def _one(self, k: int) -> bool:
        """Run one interval of k steps (or its halves, after a rollback). False: stop the run."""
        if self.rewound:  # the segment this half belonged to was rewound: nothing of it runs
            return True
        if self.on_baseline:  # a half after the other half fell back
            return self._baseline(k)
        ck = self.backend.checkpoint() if self.rollback else None
        iv = self.backend.interval(k, self.mode)
        for c in self.clauses:
            if c.kind != "rollback":
                continue
            reason = c.check(iv)
            if reason is not None and k > 1:
                self.events.append({"t": iv.t, "k": k, "clause": c.name, "action": "rollback", "why": reason})
                self.backend.restore(ck)
                if hasattr(self.controller, "observe_rollback") and not self.fixed_k:
                    self.controller.observe_rollback(iv.extra.get("change", iv.staleness), k)
                if self.never_lose and k <= self.probe_k:
                    # the smallest multirate interval failed: below it the cheap path only adds cost. This interval is
                    # redone on the baseline (exact); the run falls back for good only when such failures come often
                    # enough that the cheap path no longer pays (small_limit), not on the first one
                    self._small(True)
                    if self.small_rate > self.small_limit():
                        self.on_baseline, self.quiet_probes = True, 0
                        self.events.append({"t": iv.t, "k": k, "clause": "never-lose", "action": "baseline",
                                            "why": f"{reason}; failure rate {self.small_rate:.2f} > {self.small_limit():.2f}"})
                    return self._baseline(k)
                h = k // 2
                return self._one(h) and self._one(k - h)
        if self.never_lose and k <= self.probe_k:
            self._small(False)
        return self._accept(iv)

    def _small(self, failed: bool) -> None:
        self.small_rate += self.small_alpha * ((1.0 if failed else 0.0) - self.small_rate)

    def small_limit(self) -> float:
        """Half the failure rate at which the smallest interval stops paying, from the controller's cost model: per
        step, multirate costs mr = c_student + (1 + o_verify) / k + o_control teacher calls and the baseline base =
        1 + o_control; a failed interval wastes its own cost and is redone on the baseline, so multirate pays while the
        failure rate p < (base - mr) / base (0.44 at k 2 with the defaults)."""
        c, k = self.controller, self.probe_k
        mr = c.c_student + (1.0 + c.o_verify) / k + c.o_control
        base = 1.0 + c.o_control
        return 0.5 * max(base - mr, 0.0) / base

    def _accept(self, iv: Interval) -> bool:
        self.log.append((iv.t, iv.k, iv.staleness, iv.energy))
        if not iv.finite:
            self.events.append({"t": iv.t, "k": iv.k, "clause": "non-finite", "action": "stop"})
            return False
        for c in self.clauses:
            if c.kind != "refuse":
                continue
            reason = c.check(iv)
            if reason is not None:
                self.events.append({"t": iv.t, "k": iv.k, "clause": c.name, "action": "refuse", "why": reason})
                if not self.fixed_k:
                    step = c.step_factor() if getattr(c, "measured_step", False) and getattr(c, "rewind", False) else None
                    self.controller.refuse(iv.k, step)
                if getattr(c, "rewind", False) and not self.fixed_k and self.segment is not None:
                    ck, k_seg, n_log = self.segment
                    self.backend.restore(ck)
                    t_back = self.backend.time()
                    del self.log[n_log:]
                    for cc in self.clauses:
                        if hasattr(cc, "rewind_to"):
                            cc.rewind_to(t_back)
                    self.events.append({"t": iv.t, "k": iv.k, "clause": c.name, "action": "rewind", "why": f"to {t_back:.0f} fs"})
                    self.segment, self.rewound = None, True
                for cc in self.clauses:
                    cc.reset()  # a measured clause judges the new interval from scratch
                if self.rewound:
                    return True
                break
        if not self.fixed_k:
            # the controller takes the held quantity's change over the interval and applies its own 1/sqrt(3)
            # (the mean over a linear drift); a backend whose staleness is already that mean passes the change
            self.controller.observe(iv.extra.get("change", iv.staleness), iv.k)
        return True

    def _baseline(self, n: int) -> bool:
        """n steps of the baseline itself; the probe at its end decides whether multirate resumes."""
        probe = self.probe_k if n > self.probe_k else 0
        iv = self.backend.baseline_interval(n, probe, self.mode)
        self.baseline_steps += n
        self.log.append((iv.t, 1, iv.staleness, iv.energy))
        if not iv.finite:
            self.events.append({"t": iv.t, "k": 1, "clause": "non-finite", "action": "stop"})
            return False
        if not iv.extra.get("probed"):
            return True
        trial = Interval(t=iv.t, k=self.probe_k, staleness=iv.staleness, finite=True, energy=iv.energy, extra=iv.extra)
        reasons = [r for r in (c.check(trial) for c in self.clauses if c.kind == "rollback") if r is not None]
        self._small(bool(reasons))
        if reasons:
            self.quiet_probes = 0
            self.probe_every = min(2 * self.probe_every, self.probe_every_max)
            return True
        self.quiet_probes += 1
        if self.controller.ceiling is not None and self.controller.ceiling < self.probe_k:
            # a measured clause refused the smallest interval, and refusals latch: clean probes (rollback clauses
            # only) cannot lift that, and resuming would run multirate at k 1, dearer than the baseline itself
            return True
        if self.quiet_probes >= self.controller.quiet_to_grow:
            self.on_baseline, self.quiet_probes = False, 0
            self.controller.k = self.controller.even(self.probe_k)
            self.controller.quiet = 0
            if hasattr(self.controller, "reset_small_stats"):
                self.controller.reset_small_stats()  # the clean probes are the evidence now, not a pre-stretch average
            for c in self.clauses:
                c.resume()
            self.events.append({"t": iv.t, "k": self.probe_k, "clause": "never-lose", "action": "resume",
                                "why": f"{self.controller.quiet_to_grow} clean probes"})
        return True

    def run(self, horizon: float) -> dict:
        t0 = time.perf_counter()
        while self.backend.time() < horizon:
            left = self.backend.steps_left(horizon)
            if self.never_lose and (self.on_baseline or not self.controller.use_student()):
                if not self.on_baseline and not self._cost_fallback:
                    # the controller's own cost model says the cheap path does not pay at its interval (k 1 always):
                    # logged once per stretch, so a trace can tell it from a clause (it was silent until 2026-11-15)
                    c = self.controller
                    self.events.append({"t": self.backend.time(), "k": c.k, "clause": "cost-model",
                                        "action": "baseline", "why": f"controller k {c.k}: predicted speedup "
                                        f"{c.predicted_speedup(c.k):.2f} < {c.min_gain:.2f}"})
                    self._cost_fallback = True
                self.segment = None  # the next multirate interval checkpoints afresh, after this baseline stretch
                ok = self._baseline(min(left, self.probe_every))
            else:
                self._cost_fallback = False
                k = self._k_now(left)
                self.rewound = False
                if self.rewinding and (self.segment is None or self.segment[1] != k):
                    self.segment = (self.backend.checkpoint(), k, len(self.log))
                ok = self._one(k)
            if not ok:
                break
        self.backend.finish()
        return {"seconds": time.perf_counter() - t0, "intervals": len(self.log), "events": self.events,
                "mean_k": (sum(r[1] for r in self.log) / len(self.log)) if self.log else 0.0,
                "mode": self.mode, "baseline_steps": self.baseline_steps, "cap_bound": self.cap_bound}
