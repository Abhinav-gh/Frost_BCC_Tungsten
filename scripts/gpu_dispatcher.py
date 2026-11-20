"""GPU dispatcher: keep both GPU lanes busy until the backlog is empty (2026-11-12, user request: no GPU downtime).

The older queue was bash chains, each waiting on markers ("q_gpu0 done") before starting: a GPU sat idle whenever its
chain was waiting for the other lane's chain, or a two-GPU job held one lock while waiting for the other. This replaces
waiting with work stealing: whichever GPU frees first takes the best ready job.

Jobs: ``data/queue/jobs.json``, a list appended atomically by whoever adds work (``scripts/queue_add.py``), each
  {"name": str, "cmd": bash, "lanes": ["heavy", "gpu1"], "priority": int (lower first), "after": [job names that must
   have exited 0], "needs_files": [paths, relative to the repo, that must exist], "max_hours": float}
State: ``data/queue/state.json`` (name -> status queued / running / done / failed / blocked, lane, times, exit code).

Every POLL seconds, a lane whose lock (``/tmp/frost-<lane>-job.lock``, the one ``scripts/safe_run.sh`` takes) is free on
two consecutive polls is idle (a queued waiter would have taken it at once). The dispatcher starts the best ready job
there through ``scripts/safe_run.sh`` (which takes the lock, pins the GPU, logs to ``data/logs/jobs/<name>.log``). Jobs
must not call ``safe_run.sh`` on a GPU lane themselves (the lane lock is already theirs); CPU work goes to
``LANE=check``. The dispatcher never kills, never touches a job it did not start, and exits when nothing it owns is
queued or running. Log: ``data/logs/dispatcher.log``.

Second slots (2026-11-13, user: keep both GPUs near 100%). A job that only trains or counts (no speed measured) can share
a GPU: jobs carry "timing" (default true). When a GPU's primary job has timing false, its utilisation averaged over the
last three polls is under SHARE_UTIL and it has SHARE_FREE_GB free, the GPU's second slot (``heavy2`` on GPU 0, ``gpu1b``
on GPU 1, their own locks) starts a ready job with timing false. A timing job never starts on a GPU whose second slot is
busy, and never in a second slot: two jobs on one GPU deflate each other's speed (trap 13a). Drain for priority: when a
primary lane frees beside a busy second slot and a ready timing job outranks every ready non-timing job, the lane stays
free until the second slot ends, so non-timing work cannot keep refilling the GPU ahead of better-ranked timing work.
Short second slots (2026-11-14): while timing work is queued, a second slot takes only jobs with max_hours <= 2, and the
drain waits only for such a short second-slot job; a long one (a 1 ns run) is never waited for, the primary keeps
taking non-timing work beside it.

Usage: setsid nohup .venvs/frost-mace/bin/python scripts/gpu_dispatcher.py &      (or --dry-run: one pass, no launch)
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
Q = ROOT / "data" / "queue"
JOBS, STATE = Q / "jobs.json", Q / "state.json"
LOG = ROOT / "data" / "logs" / "dispatcher.log"
LANES = {"heavy": "/tmp/frost-heavy-job.lock", "gpu1": "/tmp/frost-gpu1-job.lock",
         "heavy2": "/tmp/frost-heavy2-job.lock", "gpu1b": "/tmp/frost-gpu1b-job.lock"}
GPU = {"heavy": "0", "gpu1": "1", "heavy2": "0", "gpu1b": "1"}
PRIMARY = {"heavy2": "heavy", "gpu1b": "gpu1"}  # second slot -> the primary lane of the same GPU
SECOND = {v: k for k, v in PRIMARY.items()}
SHARE_UTIL, SHARE_FREE_GB = 70.0, 16.0
SECOND_MAX_HOURS = 2.0  # while timing work is queued, a second slot takes only jobs this short (max_hours)
POLL = 20


def gpu_load() -> dict:
    """GPU index -> (utilisation %, free memory GB) from nvidia-smi; {} if it fails."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return {r.split(",")[0].strip(): (float(r.split(",")[1]), float(r.split(",")[2]) / 1024.0)
                for r in out.strip().splitlines() if r.count(",") == 2}
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as fh:
        fh.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC] {msg}\n")


def lock_free(path: str) -> bool:
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    except BlockingIOError:
        return False
    finally:
        os.close(fd)


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def exit_code(name: str):
    """The exit code safe_run.sh logged for ``name``, or None while it has not finished."""
    try:
        text = (ROOT / "data" / "logs" / "jobs" / f"{name}.log").read_text(errors="replace")
    except FileNotFoundError:
        return None
    m = re.findall(rf"\] {re.escape(name)} exit=(-?\d+)", text)
    return int(m[-1]) if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    Q.mkdir(parents=True, exist_ok=True)
    state = read_json(STATE, {})
    procs: dict[str, subprocess.Popen] = {}
    free_streak = dict.fromkeys(LANES, 0)
    util_hist: dict[str, list] = {"0": [], "1": []}
    draining: dict[str, bool] = {}
    mem_waiting: set[str] = set()  # timing jobs already logged as waiting for memory
    log(f"dispatcher start (pid {os.getpid()}){' dry run' if args.dry_run else ''}")
    while True:
        jobs = read_json(JOBS, [])
        for j in jobs:
            state.setdefault(j["name"], {"status": "queued"})
        # finished jobs: our own processes, or (after a restart) the exit line safe_run wrote
        for name, st in state.items():
            if st["status"] != "running":
                continue
            p = procs.get(name)
            done = (p is not None and p.poll() is not None) or (p is None and exit_code(name) is not None)
            if done:
                rc = exit_code(name)
                rc = rc if rc is not None else (p.returncode if p is not None else -1)
                st.update(status="done" if rc == 0 else "failed", exit=rc, end=time.time())
                procs.pop(name, None)
                log(f"{name} finished on {st.get('lane')}: exit {rc}")
        by_name = {j["name"]: j for j in jobs}
        for j in jobs:  # a job whose dependency failed can never run
            st = state[j["name"]]
            if st["status"] == "queued" and any(state.get(a, {}).get("status") in ("failed", "blocked")
                                                for a in j.get("after", [])):
                st["status"] = "blocked"
                log(f"{j['name']} blocked: a dependency failed")
        load = gpu_load()
        for g, (u, _) in load.items():
            util_hist.setdefault(g, []).append(u)
            del util_hist[g][:-3]

        def running_on(lane: str):
            return next((n for n, st in state.items() if st["status"] == "running" and st.get("lane") == lane), None)

        def timing_of(name) -> bool:
            return by_name.get(name, {}).get("timing", True) if name else True

        for lane, lock in LANES.items():
            free_streak[lane] = free_streak[lane] + 1 if lock_free(lock) else 0
            if free_streak[lane] < (1 if args.dry_run else 2):
                continue
            g = GPU[lane]
            if lane in PRIMARY:  # a second slot opens only beside a non-timing primary job on an underused GPU
                prim = PRIMARY[lane]
                if lock_free(LANES[prim]) or timing_of(running_on(prim)):
                    continue
                hist, free_gb = util_hist.get(g, []), load.get(g, (100.0, 0.0))[1]
                if len(hist) < 3 or sum(hist) / len(hist) >= SHARE_UTIL or free_gb < SHARE_FREE_GB:
                    continue
            second_busy = lane in SECOND and not lock_free(LANES[SECOND[lane]])
            home = PRIMARY.get(lane, lane)
            free_now = load.get(g, (0.0, 1e9))[1]

            def room(j):
                # Memory before a timing job starts (2026-11-18). A timing job waits inside for a quiet GPU
                # (wait_gpu_quiet), holding its lane: two timing runs held both lanes for 3 h while another tenant kept
                # 33-36 GB, and every untimed job (2-3 GB) queued behind them. A timing job now starts only when its
                # GPU already has its memory (min_free_gb, default 8); until then the lane takes untimed work.
                # an untimed job is held only when it declares its memory (2026-11-18: a 3,744-atom student's training
                # died twice out of memory beside the tenant's 33 GB, after a gate inside the job had passed)
                need = j.get("min_free_gb", 8.0 if j.get("timing", True) else None)
                ok = need is None or free_now >= float(need)
                if not ok and j["name"] not in mem_waiting:
                    mem_waiting.add(j["name"])
                    log(f"{j['name']} waits for {j.get('min_free_gb', 8.0)} GB free on GPU {g} (has {free_now:.1f})")
                return ok

            ready = [j for j in jobs if state[j["name"]]["status"] == "queued" and home in j.get("lanes", ["heavy", "gpu1"])
                     and all(state.get(a, {}).get("status") == "done" for a in j.get("after", []))
                     and all((ROOT / f).exists() for f in j.get("needs_files", []))
                     and (not j.get("timing", True) or (lane not in PRIMARY and not second_busy))
                     and room(j)]
            timing_queued = any(t.get("timing", True) and state[t["name"]]["status"] == "queued" for t in jobs)
            if lane in PRIMARY and timing_queued:
                # a long job in a second slot outlives its primary and then holds the GPU from every timing job for
                # hours (two 1 ns runs did, 2026-11-14): while timing work waits, second slots take short jobs only
                ready = [j for j in ready if float(j.get("max_hours", 6)) <= SECOND_MAX_HOURS]
            if not ready:
                continue
            ready.sort(key=lambda j: (j.get("priority", 50), jobs.index(j)))
            j = ready[0]
            second_job = by_name.get(running_on(SECOND[lane]) or "", {}) if lane in SECOND else {}
            if lane not in PRIMARY and second_busy and float(second_job.get("max_hours", 6)) <= SECOND_MAX_HOURS:
                # Drain for priority (2026-11-13). Beside a busy second slot only non-timing work can start; refilling
                # the GPU with it whenever one slot frees would keep a slot busy for as long as non-timing jobs are
                # queued, and every timing job (which needs the whole GPU) would starve behind lower-priority work. So
                # when a ready timing job outranks the best non-timing one, leave this lane free until the second
                # slot's job ends; the timing job then starts here.
                timed = [t for t in jobs if state[t["name"]]["status"] == "queued" and t.get("timing", True)
                         and home in t.get("lanes", ["heavy", "gpu1"])
                         and all(state.get(a, {}).get("status") == "done" for a in t.get("after", []))
                         and all((ROOT / f).exists() for f in t.get("needs_files", []))
                         and free_now >= float(t.get("min_free_gb", 8.0))]
                if timed and min(t.get("priority", 50) for t in timed) < j.get("priority", 50):
                    if not draining.get(lane):
                        log(f"{lane}: draining GPU {g} for a timing job (priority "
                            f"{min(t.get('priority', 50) for t in timed)}) until its second slot frees")
                    draining[lane] = True
                    continue
            draining[lane] = False
            cmd = (f"CUDA_VISIBLE_DEVICES={g} LANE={lane} MAX_HOURS={j.get('max_hours', 6)} scripts/safe_run.sh "
                   f"{shlex.quote(j['name'])} bash -c {shlex.quote(j['cmd'])}")
            if args.dry_run:
                log(f"would start {j['name']} on {lane}: {j['cmd'][:120]}")
                print(f"would start {j['name']} on {lane}")
                continue
            procs[j["name"]] = subprocess.Popen(["bash", "-c", cmd], cwd=ROOT, start_new_session=True,
                                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            state[j["name"]].update(status="running", lane=lane, start=time.time())
            free_streak[lane] = 0
            log(f"started {j['name']} on {lane} (GPU {g}, priority {j.get('priority', 50)}, timing {j.get('timing', True)})")
        write_json(STATE, state)
        if args.dry_run:
            return 0
        owned = [n for n in by_name if state[n]["status"] in ("queued", "running")]
        if not owned:
            log("backlog empty: dispatcher exits")
            return 0
        time.sleep(POLL)


if __name__ == "__main__":
    raise SystemExit(main())
