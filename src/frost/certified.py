"""Intervals a previous run certified, so the controller need not rediscover them.

The measured clause costs one ``DriftMonitor.min_ps`` of evidence per rung of
descent. On Si the prior starts three rungs too high, so the controller spends
60 ps descending 16 to 10 before it can run at the interval it will keep — and a
run shorter than that reports a speedup made of intervals it is in the process of
rejecting (plan section 1.14). Nothing about that descent is specific to the run:
the same system with the same student ends at the same rung.

This records the ceiling a run certified, together with what certified it, so a
later run can start there. The record is deliberately narrow. An interval is
evidence about one system with one student under one certificate, and reusing it
anywhere else is asserting a result that was never measured, so ``lookup``
matches on all of those and returns nothing when any differs.

The refusal sequence is recorded because a ceiling without it cannot be
questioned. On Si both 200 ps seeds refuse k = 16 at 20.3 ps and k = 14 at
40.3 ps identically, and the entire difference between a 4.20x entry and a 4.44x
one is whether a third refusal happens at 60.3 ps -- one rung at one timestamp.
Neither ``evidence_fs`` nor the refusal count shows that, so a later run that
inherits a ceiling could not be asked why it has the ceiling it does.

What it does not record is a pass. A ceiling means "no interval above this
survived the measured clause in that run", which is weaker than "this interval is
certified": the run may have ended while still descending, exactly as Si's 60 ps
run did. ``evidence_fs`` and ``refusals`` are stored so a caller can tell the two
apart, and ``lookup`` will not return a ceiling that was never held for at least
``min_hold_fs``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Certified:
    run: str
    student: str
    k: int
    ceiling: int
    period_fs: float
    epsilon: float
    gain_budget: float
    evidence_fs: float  # how long the run held this ceiling after its last refusal
    refusals: int
    projection: float = 0.0  # the largest projected gain seen while holding it
    refused: list = field(default_factory=list)  # [{"k", "t_fs", "projection"}, ...]

    def start_ceiling(self, scatter: float = 1.5) -> int:
        """The ceiling to inherit, demoted one rung when the margin is thin.

        Holding a ceiling is not the same as holding it comfortably. Si's two
        200 ps runs descend identically through 16 and 14 and then disagree about
        k = 12, which projects to 0.72 of the budget: one seed refused it and one
        held it for 160 ps, so the store accepted it as evidence -- correctly by
        its own rule, since it was held. Warm-starting from that entry hands one
        seed's luck to every later run, and those runs never rediscover the
        refusal because they never climb above the inherited ceiling.

        ``scatter`` is not fitted to this case. It is the seed-to-seed spread
        already measured elsewhere: the glass's two 50 ps runs of one configuration
        totalled 0.66 and 0.44 meV/atom, a factor of 1.5. It is also the only
        number in the certificate that measures *reproducibility* rather than a
        run -- epsilon, the resonance fraction and the gain budget each bound what
        one run did, while this bounds how far two runs of the same thing differ,
        which is the quantity an inherited verdict needs guarding with
        (frost-ec's observation). A ceiling is inherited
        whole only if its margin survives that much scatter; otherwise the caller
        starts one rung lower and pays one ``min_ps`` to re-earn the boundary rung
        rather than assuming it. Si's k = 10 (projection 0.53) is inherited; its
        k = 12 (0.72) is demoted to 10.
        """
        if self.projection <= 0.0 or self.projection * scatter < self.gain_budget:
            return self.ceiling
        return max(2, self.ceiling - 2 if self.ceiling > 2 else self.ceiling)

    def key(self) -> tuple:
        """What the interval is evidence about: this system, student and certificate."""
        return (self.run, self.student, round(self.period_fs, 3),
                round(self.epsilon, 6), round(self.gain_budget, 6))


def record(path: str | Path, entry: Certified) -> None:
    """Add or replace the entry for this (run, student, certificate)."""
    path = Path(path)
    rows = read(path)
    rows = [row for row in rows if row.key() != entry.key()]
    rows.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(row) for row in rows], indent=2))


def read(path: str | Path) -> list[Certified]:
    path = Path(path)
    if not path.exists():
        return []
    return [Certified(**row) for row in json.loads(path.read_text())]


def lookup(path: str | Path, run: str, student: str, period_fs: float, epsilon: float,
           gain_budget: float, min_hold_fs: float = 20_000.0) -> Certified | None:
    """The ceiling to start from, or None if nothing applies or nothing held long enough.

    ``min_hold_fs`` defaults to one ``DriftMonitor.min_ps``: a ceiling the previous
    run never held for that long was not certified by it, only reached.
    """
    wanted = (run, student, round(period_fs, 3), round(epsilon, 6), round(gain_budget, 6))
    for row in read(path):
        if row.key() == wanted and row.evidence_fs >= min_hold_fs:
            return row
    return None
