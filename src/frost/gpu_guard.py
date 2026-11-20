"""Was this timed arm alone on its GPU? (2026-11-16)

The server is shared with other tenants, invisible to our container except through the GPU: on 2026-11-15/16 another
tenant's process held 26-36 GB of each GPU and computed on both, came and went within hours, and restarted under new
process ids, so a gate at a job's start (``scripts/wait_gpu_mem.sh``) let timed runs start in a gap and then share the GPU
for part of a run (trap 13a: contention inflates or deflates a speedup silently). ``ContentionWatch`` samples, from a
background thread, the device memory in use on this job's GPU by everything except this process (NVML's device total,
minus this process's reserved memory and a context allowance), so every timed arm can carry its own record and an arm that
shared the GPU is flagged instead of quoted. NVML calls and the allocator's statistics do not enter this process's CUDA
stream, so the timed loop is not disturbed.
"""

from __future__ import annotations

import os
import threading


class ContentionWatch:
    def __init__(self, every_s: float = 2.0, context_mib: float = 800.0, threshold_mib: float = 4000.0,
                 foreign_min_mib: float = 8000.0, active_pct: float = 5.0):
        self.every_s, self.context_mib, self.threshold_mib = every_s, context_mib, threshold_mib
        self.foreign_min_mib, self.active_pct = foreign_min_mib, active_pct
        self._stop = threading.Event()
        self._thread = None
        self._last_ts = None
        self.samples: list[float] = []
        self.active: list[float] = []
        # the GPU's SM clock (MHz) and temperature (C) at each poll (2026-11-17): a certified PIMD arm run right after the
        # heavy teacher arm read 2% slower than fixed 3 doing identical work, on both seeds, with nobody else computing;
        # a hot GPU boosts less, so an arm's clocks are part of its timing record
        self.clocks: list[float] = []
        self.temps: list[float] = []
        self._handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            ids = [int(i) for i in visible.split(",") if i.strip().isdigit()]
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(ids[0] if ids else 0)
            self._nvml = pynvml
        except Exception:  # noqa: BLE001  (no NVML: record nothing rather than fail a run)
            self._handle = None

    def _foreign_mib(self) -> float:
        import torch

        used = self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used / 2**20
        own = torch.cuda.memory_reserved() / 2**20
        return max(0.0, used - own - self.context_mib)

    def _foreign_active(self) -> float:
        """The largest SM utilisation (%) of a foreign process since the last poll: a process that is not this one
        (this one: the process whose GPU memory is nearest this process's reserved memory) and holds more than
        ``foreign_min_mib`` (our own helper servers hold 1-2 GB; the tenant held 33-36 GB). Memory alone over-flags: on
        2026-11-16 the tenant sat on both GPUs at 0% while our timings matched the quiet benchmark."""
        import torch

        nv = self._nvml
        own = torch.cuda.memory_reserved() / 2**20 + 500.0
        procs = [(p.pid, (p.usedGpuMemory or 0) / 2**20) for p in nv.nvmlDeviceGetComputeRunningProcesses(self._handle)]
        near = min(procs, key=lambda q: abs(q[1] - own)) if procs else None
        mine = near[0] if near is not None and abs(near[1] - own) < 4000.0 else None
        foreign = {pid for pid, mib in procs if pid != mine and mib > self.foreign_min_mib}
        if not foreign:
            self._last_ts = None
            return 0.0
        ts = self._last_ts if self._last_ts is not None else int((__import__("time").time() - self.every_s) * 1e6)
        try:
            samples = nv.nvmlDeviceGetProcessUtilization(self._handle, ts)
        except nv.NVMLError:
            return 0.0
        self._last_ts = max([x.timeStamp for x in samples], default=ts)
        return float(max([x.smUtil for x in samples if x.pid in foreign], default=0))

    def _clock_temp(self) -> None:
        nv = self._nvml
        self.clocks.append(float(nv.nvmlDeviceGetClockInfo(self._handle, nv.NVML_CLOCK_SM)))
        self.temps.append(float(nv.nvmlDeviceGetTemperature(self._handle, nv.NVML_TEMPERATURE_GPU)))

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(self._foreign_mib())
                self.active.append(self._foreign_active())
            except Exception:  # noqa: BLE001
                pass
            try:
                self._clock_temp()
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(self.every_s)

    def start(self) -> "ContentionWatch":
        self.samples, self.active, self._stop, self._last_ts = [], [], threading.Event(), None
        self.clocks, self.temps = [], []
        if self._handle is not None:
            self.samples.append(self._foreign_mib())
            try:
                self.active.append(self._foreign_active())
            except Exception:  # noqa: BLE001
                pass
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> dict:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2 * self.every_s + 1)
            try:
                self.samples.append(self._foreign_mib())
            except Exception:  # noqa: BLE001
                pass
        if not self.samples:
            return {"foreign_mib_max": None, "contended": None, "samples": 0}
        mx = max(self.samples)
        # the share of samples with foreign memory present: another tenant came and went within single arms
        # (2026-11-16: its memory was present at some sample of every arm of a run whose baseline still matched the quiet
        # benchmark), so the maximum alone cannot tell a brief or idle visit from a shared arm
        frac = sum(v > self.threshold_mib for v in self.samples) / len(self.samples)
        act = [a for a in self.active if a is not None]
        busy = sum(a > self.active_pct for a in act) / len(act) if act else None
        out = {"foreign_mib_max": round(mx, 1), "foreign_mib_mean": round(sum(self.samples) / len(self.samples), 1),
               "samples": len(self.samples), "contended_fraction": round(frac, 3),
               "foreign_sm_max": max(act) if act else None,
               "foreign_active_fraction": round(busy, 3) if busy is not None else None}
        # contended = another process computed on this GPU during the arm (activity, when NVML reports it; memory
        # presence alone otherwise)
        out["contended"] = bool(busy > 0) if busy is not None else bool(mx > self.threshold_mib)
        if self.clocks:
            out["sm_clock_mhz_mean"] = round(sum(self.clocks) / len(self.clocks), 1)
            out["sm_clock_mhz_min"] = min(self.clocks)
        if self.temps:
            out["temp_c_mean"] = round(sum(self.temps) / len(self.temps), 1)
            out["temp_c_max"] = max(self.temps)
        return out
