"""Certified intervals carried across runs of one configuration (2026-11-14).

Under a thermostat each rung past the NVE-validated prior waits for its own evidence (ThermostatBias.min_ps), so a run
spends its first ~5 ps at the prior before the first admission even when the same system, student, teacher and
certificate were certified in an earlier run (copper: k 24 for 5 ps, then 48 and 50; a fixed 48 skips that and wins a
20 ps race by 8.6% while losing every longer one). A warm start begins where the last certified run of the identical
configuration ended: its interval, its admitted probation rung and its ceiling. Nothing is skipped in the judging: the
measured clauses start from an empty record, refuse and latch as always, and the arm's JSON names the record it started
from, so a warm number is never quoted as a cold one.

The key is a hash of everything the certificate depends on (system and its temperature, the student file and its
modification time, the teacher, ensemble and thermostat, the step and masses, the period, epsilon, the search bound, the
lattice, k_max and the clause set). The store is one JSON file shared by concurrent runs (CLAUDE.md trap 20): updates take
an exclusive lock on a sibling lock file and replace the file atomically.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from pathlib import Path


def config_key(config: dict) -> str:
    """A stable hash of a configuration (keys sorted, floats as written by json)."""
    return hashlib.sha1(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:16]


def load(path: str | os.PathLike, key: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p.with_suffix(p.suffix + ".lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_SH)
        data = json.loads(p.read_text()) if p.stat().st_size else {}
    return data.get(key)


def save(path: str | os.PathLike, key: str, record: dict) -> None:
    """Write (replace) the record for ``key``, under an exclusive lock, atomically."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p.with_suffix(p.suffix + ".lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        data = json.loads(p.read_text()) if p.exists() and p.stat().st_size else {}
        data[key] = dict(record, saved=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
        tmp = p.with_suffix(p.suffix + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(data, indent=1, sort_keys=True))
        tmp.replace(p)


def warm_start(controller, record: dict) -> dict:
    """Start ``controller`` from a stored record: its interval, probation rung and ceiling, each clipped by this run's own
    bounds (k_max, the search bound). Returns what was applied, for the arm's JSON."""
    bound = min(controller.k_max, controller.resonance_cap())
    applied = {}
    if record.get("probation") is not None and controller.probation is not None:
        controller.probation = max(controller.probation, min(int(record["probation"]), bound))
        applied["probation"] = controller.probation
    if record.get("ceiling") is not None:
        controller.ceiling = min(int(record["ceiling"]), bound)
        applied["ceiling"] = controller.ceiling
    k = int(record.get("k", controller.k))
    cap = min(bound, controller.ceiling or bound, controller.probation or bound)
    controller.k = controller.even(max(1, min(k, cap)))
    applied["k"] = controller.k
    return applied
