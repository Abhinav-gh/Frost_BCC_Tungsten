"""Start a timed job only when its GPU is quiet (2026-11-16).

    python scripts/wait_gpu_quiet.py MIN_FREE_GB QUIET_MIN MAX_MIN -- CMD ...

Waits until this job's GPU (CUDA_VISIBLE_DEVICES, a physical index, as the dispatcher sets it) has MIN_FREE_GB GiB free
AND no other process on it has computed (SM utilisation above 5%, from NVML's per-process samples) for QUIET_MIN
minutes, then replaces itself with CMD (exec). Gives up with exit 3 after MAX_MIN minutes. The earlier gate
(``wait_gpu_mem.sh``, free memory only) waited for hours behind another tenant's idle process (35 GB held at 0%) and could
not tell an idle visitor from a busy one; what spoils a timing is the other process's compute, and the per-arm record
(``frost.gpu_guard.ContentionWatch``) still catches a return mid-run.
"""

from __future__ import annotations

import os
import sys
import time


def main(argv: list[str]) -> int:
    if "--" not in argv:
        raise SystemExit(__doc__)
    i = argv.index("--")
    min_free_gb, quiet_min, max_min = (float(v) for v in argv[:i])
    cmd = argv[i + 1:]
    import pynvml

    pynvml.nvmlInit()
    gpu = int((os.environ.get("CUDA_VISIBLE_DEVICES", "0") or "0").split(",")[0])
    h = pynvml.nvmlDeviceGetHandleByIndex(gpu)
    t0 = time.time()
    quiet_since, last_ts, said = None, int((t0 - 5) * 1e6), False
    while time.time() - t0 < 60 * max_min:
        free_gb = pynvml.nvmlDeviceGetMemoryInfo(h).free / 2**30
        try:
            samples = pynvml.nvmlDeviceGetProcessUtilization(h, last_ts)
            last_ts = max([s.timeStamp for s in samples], default=last_ts)
            busy = max([s.smUtil for s in samples], default=0) > 5
        except pynvml.NVMLError:  # no samples since last_ts: nothing computed
            busy = False
        now = time.time()
        if busy or free_gb < min_free_gb:
            quiet_since = None
        elif quiet_since is None:
            quiet_since = now
        if quiet_since is not None and now - quiet_since >= 60 * quiet_min:
            if said:
                print(f"wait_gpu_quiet: GPU {gpu} quiet with {free_gb:.1f} GiB free after {(now - t0) / 60:.0f} min",
                      flush=True)
            os.execvp(cmd[0], cmd)
        if not said:
            print(f"wait_gpu_quiet: GPU {gpu}: {free_gb:.1f} GiB free, busy={busy}; waiting for {min_free_gb:g} GiB and "
                  f"{quiet_min:g} quiet minutes", flush=True)
            said = True
        time.sleep(15)
    print(f"wait_gpu_quiet: GPU {gpu} never quiet in {max_min:g} min", flush=True)
    return 3


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
