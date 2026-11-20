"""Run a timed job until every arm ran alone on its GPU (2026-11-17).

    python scripts/clean_timing.py RESULT_JSON MAX_TRIES -- CMD ...

Another tenant holds 33-36 GB of each GPU and computes in bursts: on 2026-11-17 it passed the start gate
(``wait_gpu_quiet.py``: 2 quiet minutes) and then computed through every arm of two 33-minute PIMD runs (18-82% of each
arm's samples), so neither could be quoted. A gate at the start cannot see the future; the per-arm record
(``frost.gpu_guard.ContentionWatch``, ``arms.<name>.contention.contended``) can see the past. This wrapper runs the job
behind the quiet gate, reads the result, and while any arm shared its GPU keeps that attempt's file (renamed
``<stem>_sharedN.json``, for the record) and runs again, up to MAX_TRIES. Short timing jobs fit the tenant's quiet
windows; the science (fidelity) comes from the long runs, whose observables contention does not change.
An attempt that dies out of memory (the GPU's memory taken after the gate) is retried the same way.
Exit 0: a clean result at RESULT_JSON; 3: the gate timed out; 4: no clean attempt in MAX_TRIES.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def shared_arms(result: Path) -> list[str]:
    arms = json.loads(result.read_text()).get("arms", {})
    return [name for name, arm in arms.items() if (arm.get("contention") or {}).get("contended")]


def main(argv: list[str]) -> int:
    if "--" not in argv or argv.index("--") != 2:
        raise SystemExit(__doc__)
    result, tries, cmd = Path(argv[0]), int(argv[1]), argv[3:]
    if not result.is_absolute():
        result = ROOT / result
    for attempt in range(1, tries + 1):
        # FROST_MIN_FREE_GB: a job larger than the default 8 GiB (PIMD on 22,000 bead-atoms needs ~17) waits for its room
        min_free = os.environ.get("FROST_MIN_FREE_GB", "8")
        proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "wait_gpu_quiet.py"), min_free, "2", "600", "--",
                               *cmd], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        sys.stdout.write(proc.stdout)
        sys.stdout.flush()
        rc = proc.returncode
        if rc != 0 and ("CUDA out of memory" in proc.stdout or "OutOfMemoryError" in proc.stdout):
            # 2026-11-17: the tenant allocated 37 GB after the gate had seen 20 GB free, and the run died building its
            # graphs. An out-of-memory exit is the GPU being taken, like a shared attempt, not a fault of the run
            print(f"clean_timing: attempt {attempt}: out of memory (the GPU's memory was taken after the gate); "
                  f"waiting for a quiet GPU again", flush=True)
            continue
        if rc != 0:
            print(f"clean_timing: attempt {attempt} exited {rc}", flush=True)
            return rc
        shared = shared_arms(result)
        if not shared:
            print(f"clean_timing: attempt {attempt}: every arm ran alone on its GPU -> {result}", flush=True)
            return 0
        keep = result.with_name(f"{result.stem}_shared{attempt}.json")
        result.replace(keep)
        print(f"clean_timing: attempt {attempt}: arms {', '.join(shared)} shared the GPU (kept as {keep.name}); "
              f"waiting for a quiet GPU again", flush=True)
    print(f"clean_timing: no clean attempt in {tries}", flush=True)
    return 4


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
