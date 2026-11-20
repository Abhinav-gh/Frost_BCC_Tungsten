"""Append jobs to the GPU dispatcher's list (``data/queue/jobs.json``; ``scripts/gpu_dispatcher.py``), atomically.

Usage:
  python scripts/queue_add.py NAME "bash command" [--priority 10] [--hours 6] [--after JOB ...] [--needs FILE ...]
                                                  [--lanes heavy gpu1]
The command runs inside a job that already holds a GPU lane: it must not call scripts/safe_run.sh on a GPU lane (CPU
scoring goes to LANE=check). A name already in the list is refused. If the dispatcher has exited (backlog was empty),
start it again: setsid nohup .venvs/frost-mace/bin/python scripts/gpu_dispatcher.py &
"""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "data" / "queue" / "jobs.json"
ENV = "cd /home/shashank/Frost && source scripts/env.sh mace >/dev/null 2>&1 && P=.venvs/frost-mace/bin/python && "


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("cmd")
    ap.add_argument("--priority", type=int, default=10)
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--after", nargs="*", default=[])
    ap.add_argument("--needs", nargs="*", default=[])
    ap.add_argument("--lanes", nargs="*", default=["heavy", "gpu1"])
    ap.add_argument("--no-timing", action="store_true",
                    help="the job measures no speed (training, preparation, counting): it may share a GPU (second slots)")
    ap.add_argument("--min-free-gb", type=float, default=None,
                    help="a timing job starts only when its GPU already has this much memory free (dispatcher default 8)")
    a = ap.parse_args()
    JOBS.parent.mkdir(parents=True, exist_ok=True)
    with open(JOBS.with_suffix(".lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        jobs = json.loads(JOBS.read_text()) if JOBS.exists() else []
        if any(j["name"] == a.name for j in jobs):
            raise SystemExit(f"{a.name} is already queued")
        job = {"name": a.name, "cmd": ENV + a.cmd, "lanes": a.lanes, "priority": a.priority, "max_hours": a.hours,
               "after": a.after, "needs_files": a.needs}
        if a.no_timing:
            job["timing"] = False
        if a.min_free_gb is not None:
            job["min_free_gb"] = a.min_free_gb
        jobs.append(job)
        tmp = JOBS.with_suffix(".tmp")
        tmp.write_text(json.dumps(jobs, indent=1))
        tmp.replace(JOBS)
    print(f"queued {a.name} (priority {a.priority}); {len(jobs)} jobs in the list")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
