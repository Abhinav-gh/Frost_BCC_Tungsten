"""Choose the execution mode and interval from profiled costs and the clauses' priors.

Costs per call, in one unit (ms): ``c_t`` the expensive operator, ``c_s`` one cheap step
(student step, or solver step), ``o`` per-interval overhead. Per k cheap steps:

    sync:        k c_s + c_t + o                (one device)
    lag:         max(k c_s, c_t) + o            (two devices; MD: NVT only, CFD: refused at 3D)
    spec:        2 k c_s + c_t + o              (one device; the look-ahead doubles the cheap work)
    spec, two devices with a cheap student: as sync, but the ramp lifts the resonance prior
    extrap (CFD): k c_s + c_t + o               (free; changes accuracy, not cost)

The baseline is ``c_t`` per step (the expensive operator every step), so the speedup of a mode
at interval k is k c_t / cost. The planner returns every admissible (mode, k) with its predicted
speedup, best first; the clauses then certify (or refuse) the choice online.
"""

from __future__ import annotations


def cost(mode: str, k: int, c_t: float, c_s: float, o: float = 0.0) -> float:
    if mode in ("sync", "extrap"):
        return k * c_s + c_t + o
    if mode == "lag":
        return max(k * c_s, c_t) + o
    if mode == "spec":
        return 2 * k * c_s + c_t + o
    raise ValueError(mode)


def choose_student(replays: dict, rms_inv_mass: float, resonance_cap: int, epsilon: float = 2.0,
                   teacher_ms: float = 1.0, overhead_ms: float = 0.4) -> list:
    """Rank candidate students from their replays, before any live run (MD).

    ``replays``: name -> {"hold_R": {k: RMS residual change over k steps, eV/A}, "student_ms": the
    student's force call, ms}. The staleness at k in the certificate's units is
    hold_R(k) * 1e3 * RMS(1/m) (meV/(A amu)); the largest k under ``epsilon`` (log-interpolated
    between measured k) and under the resonance prior is the interval the certificate should admit,
    and the predicted speedup is teacher / (student + overhead + teacher / k) per step. Returns
    [(speedup, name, k)], best first.
    """
    import math

    out = []
    for name, r in replays.items():
        ks = sorted(int(k) for k in r["hold_R"])
        stale = {k: float(r["hold_R"][str(k)] if str(k) in r["hold_R"] else r["hold_R"][k]) * 1e3 * rms_inv_mass
                 for k in ks}
        k_mag = 1.0
        for a, b in zip(ks, ks[1:]):
            if stale[b] <= epsilon:
                k_mag = float(b)
                continue
            if stale[a] <= epsilon < stale[b]:  # log-log interpolation inside [a, b]
                t = math.log(epsilon / stale[a]) / math.log(stale[b] / stale[a])
                k_mag = math.exp(math.log(a) + t * math.log(b / a))
            break
        k = max(1, min(int(k_mag), resonance_cap))
        c = (r["student_ms"] + overhead_ms) / teacher_ms
        out.append((1.0 / (c + 1.0 / k), name, k))
    return sorted(out, reverse=True)


def plan(c_t: float, c_s: float, caps: dict, o: float = 0.0, devices: int = 1, nve: bool = True) -> list:
    """``caps``: mode -> largest admissible k (from the priors). Returns [(speedup, mode, k)]."""
    out = []
    for mode, kmax in caps.items():
        if mode == "lag" and (devices < 2 or nve):
            continue
        for k in range(1, int(kmax) + 1):
            out.append((k * c_t / cost(mode, k, c_t, c_s, o), mode, k))
    return sorted(out, reverse=True)
