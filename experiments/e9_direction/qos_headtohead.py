"""N2: Frost's certificate against approximate-computing quality monitors, on recorded runs.

The closest systems prior art for "skip work, check quality online, back off" is approximate
computing with runtime quality control: Green (Baek and Chilimbi, PLDI 2010) samples the output
error of an approximation at run time and backs off when it exceeds a QoS threshold; Rumba
(Khudia et al., ISCA 2015) checks individual outputs and re-executes the ones whose error is
large. Their multirate analogues are a threshold on the sampled output error at each refresh
(the residual's staleness: Green) and a threshold on the worst refresh (Rumba). Theorem B says
no estimator built from refresh-rate samples can bound impulse-multirate error, so both should
fail to separate the runs that keep the teacher's physics from the ones that do not, at *any*
threshold, while Frost's certificate (magnitude, resonance prior, drift posterior) separates them.

Ground truth is G3 on every 50 ps NVE run on disk, judged by the project's own criteria, never by
a monitor: a run fails if its energy gain by 50 ps reaches 1 meV/atom, or if its RDF difference
from the teacher exceeds 1.5x the teacher's own half-to-half spread (Cu at k = 32 fails on
structure, not energy). Every monitor is then replayed on the same runs:

- Green, oracle-calibrated: the best single threshold on the mean refresh error, chosen *after
  seeing every verdict* (the most a calibrated QoS monitor could do), in acceleration units and in
  raw force units;
- Green, per-system calibrated: the best threshold fitted per system, again on the verdicts;
- Rumba: the same on the worst refresh;
- Frost: magnitude (mean staleness <= 2.0 meV/(A amu)) and resonance (k dt <= 0.22 T on the even
  lattice, periods from the teacher's own spectrum) as priors, and ``DriftMonitor``'s projection
  replayed on the run's 1 ps energy trace as the posterior, with its shipped settings.

No GPU; reads ``data/results/e6/fidelity_*_nve_*.json``.

Output: data/results/e9/qos_headtohead.json
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from frost.controller import DriftMonitor, IntervalController

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "results" / "e9" / "qos_headtohead.json"
PERIOD_FS = {"silica1152": 27.7, "silica_glass1152": 28.4, "li3po4": 33.0, "si512": 78.7, "cu500": 113.7}
RDF_FACTOR = 1.5  # structure fails when the RDF difference exceeds this many own-spreads
# G3 as the plan writes it is energy plus structure; the structural half rests on a half-to-half
# spread that follows the seed (plan 1.16: admissible arms span 0.0023-0.0135), so every result is
# also reported against the energy line alone
TRUTH = "energy+structure"


def system_of(name: str) -> str:
    for key in sorted(PERIOD_FS, key=len, reverse=True):
        if name.startswith(key):
            return key
    raise KeyError(name)


def load_runs() -> list[dict]:
    runs = []
    for path in sorted(glob.glob(str(ROOT / "data" / "results" / "e6" / "fidelity_*_nve_*.json"))):
        try:
            d = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        cfg, fr, st = d.get("config", {}), d.get("frost"), d.get("staleness")
        if not fr or st is None or cfg.get("fs", 0) < 50000 or cfg.get("respa") not in (None, "boundary"):
            continue
        trace = fr.get("energy_trace_meV_per_atom_1ps") or []
        if len(trace) < 50:
            continue
        name = Path(path).stem.removeprefix("fidelity_")
        comp, own = d.get("comparison") or {}, d.get("teacher_self") or {}
        rdf, rdf_own = comp.get("rdf_l1_relative"), own.get("rdf_l1_relative")
        gain = float(trace[min(50, len(trace) - 1)])
        structure_fail = (TRUTH == "energy+structure" and rdf is not None and rdf_own is not None
                          and np.isfinite(rdf) and np.isfinite(rdf_own) and rdf > RDF_FACTOR * rdf_own)
        runs.append({
            "name": name, "system": system_of(name), "k": int(cfg["k"]), "dt_fs": float(cfg["dt"]),
            "mean_accel": float(st.get("mean_meV_per_A_amu", st["mean"])),
            "max_accel": float(st.get("max_meV_per_A_amu", st["max"])),
            "mean_force": float(st.get("mean_meV_per_A", float("nan"))),
            "gain_50ps": gain, "rdf_l1": rdf, "rdf_own": rdf_own, "trace": [float(x) for x in trace],
            "fails": bool(gain >= 1.0 or structure_fail),
            "why": ("energy" if gain >= 1.0 else "structure" if structure_fail else "pass"),
        })
    return runs


def best_threshold(values: np.ndarray, fails: np.ndarray) -> dict:
    """The threshold (admit at or below it) with the fewest errors, chosen on the verdicts."""
    cands = np.unique(np.concatenate([values, [values.min() - 1.0, values.max() + 1.0]]))
    best = None
    for t in cands:
        admit = values <= t
        false_admit = int((admit & fails).sum())
        false_refuse = int((~admit & ~fails).sum())
        row = {"threshold": float(t), "false_admits": false_admit, "false_refusals": false_refuse,
               "errors": false_admit + false_refuse}
        if best is None or row["errors"] < best["errors"]:
            best = row
    return best


def frost_verdict(run: dict) -> dict:
    ctrl = IntervalController(period_fs=PERIOD_FS[run["system"]], dt_fs=run["dt_fs"], k_max=64)
    cap = ctrl.resonance_cap()
    magnitude_ok = run["mean_accel"] <= ctrl.epsilon
    resonance_ok = run["k"] <= cap
    mon = DriftMonitor()
    refused_at = None
    for b, e in enumerate(run["trace"]):
        if mon.over_projection(e, 1000.0 * b):
            refused_at = b
            break
    admitted = magnitude_ok and resonance_ok and refused_at is None
    return {"cap": cap, "magnitude_ok": magnitude_ok, "resonance_ok": resonance_ok,
            "drift_refused_at_ps": refused_at, "admits": admitted}


def main() -> int:
    global TRUTH
    full = {}
    for truth in ("energy+structure", "energy"):
        TRUTH = truth
        full[truth] = evaluate()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(full, indent=2))
    print(f"-> {OUT}")
    return 0


def evaluate() -> dict:
    runs = load_runs()
    fails = np.array([r["fails"] for r in runs])
    report = {"truth": TRUTH, "n_runs": len(runs), "n_fail": int(fails.sum()), "rdf_factor": RDF_FACTOR,
              "monitors": {}}
    systems = sorted({r["system"] for r in runs})

    for label, key in (("green_accel", "mean_accel"), ("green_force", "mean_force"),
                       ("rumba_accel", "max_accel")):
        vals = np.array([r[key] for r in runs])
        ok = np.isfinite(vals)
        report["monitors"][label + "_oracle"] = best_threshold(vals[ok], fails[ok])
        per = {}
        for sysname in systems:
            idx = np.array([r["system"] == sysname for r in runs]) & ok
            if idx.sum():
                per[sysname] = best_threshold(vals[idx], fails[idx])
        report["monitors"][label + "_per_system_oracle"] = {
            "errors": int(sum(v["errors"] for v in per.values())),
            "false_admits": int(sum(v["false_admits"] for v in per.values())),
            "false_refusals": int(sum(v["false_refusals"] for v in per.values())), "per_system": per}
        # deployment: calibrated offline on the other systems, run on a new one (how a QoS monitor
        # is calibrated, by profiling training inputs)
        fa = fr = 0
        for sysname in systems:
            held = np.array([r["system"] == sysname for r in runs]) & ok
            train = ~held & ok
            if not held.sum() or not train.sum():
                continue
            t = best_threshold(vals[train], fails[train])["threshold"]
            admit = vals[held] <= t
            fa += int((admit & fails[held]).sum())
            fr += int((~admit & ~fails[held]).sum())
        report["monitors"][label + "_leave_one_system_out"] = {"errors": fa + fr, "false_admits": fa,
                                                               "false_refusals": fr}

    rows = []
    fa = fr = 0
    for r in runs:
        v = frost_verdict(r)
        fa += int(v["admits"] and r["fails"])
        fr += int((not v["admits"]) and not r["fails"])
        rows.append({k: r[k] for k in ("name", "system", "k", "dt_fs", "mean_accel", "max_accel",
                                        "gain_50ps", "rdf_l1", "rdf_own", "fails", "why")} | v)
    report["monitors"]["frost"] = {"false_admits": fa, "false_refusals": fr, "errors": fa + fr}
    report["runs"] = rows

    print(f"\n=== truth: {TRUTH} === {len(runs)} runs of 50 ps NVE, {int(fails.sum())} fail")
    for name, m in report["monitors"].items():
        print(f"  {name:32s} errors {m['errors']:2d} (false admits {m['false_admits']}, "
              f"false refusals {m['false_refusals']})"
              + (f" at threshold {m['threshold']:.3f}" if "threshold" in m else ""))
    print("\nrun                                            k   mean  gain50  truth      frost")
    for r in rows:
        why = ("admit" if r["admits"] else "refuse:" + ",".join(
            c for c, okc in (("mag", r["magnitude_ok"]), ("res", r["resonance_ok"]),
                             ("drift", r["drift_refused_at_ps"] is None)) if not okc))
        print(f"  {r['name'][:44]:44s} {r['k']:3d} {r['mean_accel']:6.3f} {r['gain_50ps']:7.3f}  "
              f"{r['why']:9s}  {why}")
    return report


if __name__ == "__main__":
    raise SystemExit(main())
