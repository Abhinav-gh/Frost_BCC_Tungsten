"""Week 1 wrap-up: pick the best student from the silica sweep, then run it online.

Selection rule. The rule is the objective itself. For every configuration of
the sweep (data/results/e5/student_replay_<tag>.json, run silica1152_T300)
compute:
- k*, the largest k in {1, 2, 4, 8, 16} whose measured mean staleness h_R(k) on
  unseen silica windows is within --rank-epsilon. This ranking stays in force
  units because the replay statistics are stored that way; the runs themselves are
  certified in acceleration units (--units), which is what decides k online;
- c_S, the CUDA-graph student call over the cuEquivariance teacher call.

The predicted speedup is S = 1 / (c_S + 1.02 / k* + 0.01) (note Eq. 8), or 1 when
k* = 1. The configuration with the largest S wins; ties go to the lower h_R(4).
Only configurations whose student captured in a CUDA graph and whose teacher ran
with cuEquivariance are eligible, because those are what the online loop uses.

Runs. The winner goes through experiments/e5_student/online.py against the
cuEquivariance teacher (baseline B0):
- every run in --cold from a cold start (default Si512 and silica at 300 K),
  training in situ during a 500 fs bootstrap and freezing at 1500 fs;
- every run in --warm from its replay weights, frozen from the start (default
  silica; needs data/models/students/<run>_<tag>.pt).

--online-args passes kernel options to every run, for example
"--teacher-graph --student-backend cueq --neighbours alchemi". The frozen-phase
speedup comes from the wall-clock stamps in the online log, so training cost and
multirate speed are reported separately.

Output: data/results/e5/week1_online_summary[_<tag>].json

Usage (guarded launcher; queues behind the sweep)
  scripts/safe_run.sh e5_week1_online .venvs/frost-mace/bin/python \\
      experiments/e5_student/week1_online.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "data" / "results" / "e5"
STUDENTS = ROOT / "data" / "models" / "students"
TAGS = ("f5c32l2", "f5c32l3", "f5c32l3d4", "m5c32", "m4c16")
SELECT_RUN = "silica1152_T300"


def score(entry: dict, epsilon_mev: float, student_ms: float | None = None,
          teacher_ms: float | None = None) -> dict | None:
    """Predicted speedup of one sweep entry; timings default to the replay's own."""
    timing = entry.get("timing", {})
    if student_ms is None or teacher_ms is None:
        if "student_graph_ms" not in timing or "teacher_cueq_force_ms" not in timing:
            return None
        student_ms, teacher_ms = timing["student_graph_ms"], timing["teacher_cueq_force_ms"]
    c_s = student_ms / teacher_ms
    hold = entry["frost_student"]["hold_R"]
    k_star = 1
    for k in (2, 4, 8, 16):
        h = hold.get(str(k), hold.get(k))
        if h is not None and h * 1e3 <= epsilon_mev:
            k_star = k
    speedup = 1.0 if k_star == 1 else 1.0 / (c_s + 1.02 / k_star + 0.01)
    return {
        "c_S_vs_cueq": c_s,
        "k_star": k_star,
        "predicted_speedup": speedup,
        "hold_R4_meV": hold.get("4", hold.get(4)) * 1e3,
        "level_meV": entry["frost_student"]["level"] * 1e3,
        "acceptance16": entry["frost_student"]["energy"]["16"]["acceptance_proxy"],
        "student": entry["student"],
    }


def student_args(cfg: dict) -> list[str]:
    kind = cfg.get("kind", "frost")
    out = ["--student", kind, "--cutoff", str(cfg["cutoff"]), "--channels", str(cfg["channels"])]
    if kind == "frost":
        out += ["--mix", str(cfg["mix"]), "--hidden", str(cfg["hidden"]),
                "--lmax", str(cfg.get("lmax", 2))]
    else:
        out += ["--max-ell", str(cfg["max_ell"]), "--layers", str(cfg["layers"]),
                "--correlation", str(cfg["correlation"])]
    return out


def frozen_speedup(data: dict) -> dict:
    """Wall time per fs after the student froze, from the per-outer-step stamps."""
    freeze = data.get("frozen_at_fs", data["config"]["freeze_after_fs"])
    rows = [r for r in data.get("log", []) if r["phase"] == "multirate" and r["t_fs"] > freeze]
    if len(rows) < 2 or "wall_s" not in rows[0]:
        return {}
    ms_per_fs = (rows[-1]["wall_s"] - rows[0]["wall_s"]) / (rows[-1]["t_fs"] - rows[0]["t_fs"]) * 1e3
    return {"frozen_ms_per_fs": ms_per_fs,
            "speedup_frozen_phase": data["baseline_ms_per_fs"] / ms_per_fs}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rank-epsilon", type=float, default=20.0,
                    help="force-unit budget (meV/A) used only to rank sweep students here; "
                         "the run's own certificate is set by --units inside online.py")
    ap.add_argument("--units", choices=("acceleration", "force"), default="acceleration",
                    help="certificate units for the runs")
    ap.add_argument("--fs", type=int, default=3000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cold", default="si512_T300,silica1152_T300",
                    help="comma-separated E1 runs to start from a cold student")
    ap.add_argument("--warm", default=SELECT_RUN,
                    help="comma-separated E1 runs to start from their replay student")
    ap.add_argument("--online-args", default="", help="extra arguments for every online.py run")
    ap.add_argument("--tag", default="", help="suffix for the run tags and the summary file")
    ap.add_argument("--cost-file", default="",
                    help="student_cost JSON whose call times replace the replay's, "
                         "e.g. data/results/e5/student_cost_silica1152_T300_v3.json")
    ap.add_argument("--cost-key", default="graph_cueq_ms",
                    help="which student call in --cost-file to score")
    ap.add_argument("--select", default="",
                    help="run this sweep tag instead of the best-predicted one (recorded in the summary)")
    args = ap.parse_args()

    cost = json.loads(Path(args.cost_file).read_text()) if args.cost_file else None
    candidates = {}
    for tag in TAGS:
        path = RESULTS / f"student_replay_{tag}.json"
        entry = json.loads(path.read_text()).get("runs", {}).get(SELECT_RUN) if path.exists() else None
        if entry and cost:
            row = cost["students"].get(f"{SELECT_RUN}_{tag}", {})
            scored = (score(entry, args.rank_epsilon, row[args.cost_key],
                            cost["c_S_denominator_ms"]) if args.cost_key in row else None)
        else:
            scored = score(entry, args.rank_epsilon) if entry else None
        if scored:
            candidates[tag] = scored
    if not candidates:
        print("no sweep configuration has both a CUDA-graph student and a cuEq teacher time",
              flush=True)
        return 2
    best = max(candidates, key=lambda t: (candidates[t]["predicted_speedup"],
                                          -candidates[t]["hold_R4_meV"]))
    if args.select:
        if args.select not in candidates:
            print(f"--select {args.select} is not an eligible sweep tag: {sorted(candidates)}",
                  flush=True)
            return 2
        best = args.select
    for tag, c in candidates.items():
        print(f"{tag}: S={c['predicted_speedup']:.2f} k*={c['k_star']} c_S={c['c_S_vs_cueq']:.3f} "
              f"h_R(4)={c['hold_R4_meV']:.1f} meV/A{'  <- selected' if tag == best else ''}",
              flush=True)

    script = str(ROOT / "experiments" / "e5_student" / "online.py")
    common = ["--device", args.device, "--teacher-cueq", "--units", args.units,
              "--fs", str(args.fs), "--baseline-fs", "300", *shlex.split(args.online_args)]
    cfg = candidates[best]["student"]
    suffix = f"_{args.tag}" if args.tag else ""
    jobs = [(run, f"{best}_cold{suffix}",
             ["--bootstrap-fs", "500", "--freeze-after-fs", "1500", *student_args(cfg)])
            for run in filter(None, args.cold.split(","))]
    jobs += [(run, f"{best}_warm{suffix}",
              ["--bootstrap-fs", "50", "--freeze-after-fs", "0", "--train-per-call", "0",
               "--warm-start", str(STUDENTS / f"{run}_{best}.pt")])
             for run in filter(None, args.warm.split(","))]
    summary = {"rank_epsilon_meV_per_A": args.rank_epsilon, "units": args.units,
               "selection_run": SELECT_RUN, "candidates": candidates,
               "selected": best, "online_args": args.online_args, "cost_file": args.cost_file,
               "cost_key": args.cost_key if args.cost_file else None,
               "selected_by": "--select" if args.select else "predicted speedup", "runs": {}}
    out_path = RESULTS / f"week1_online_summary{suffix}.json"
    for run, tag, extra in jobs:
        rc = subprocess.call([sys.executable, script, "--run", run, "--tag", tag, *common, *extra],
                             cwd=ROOT)
        result = RESULTS / f"online_{run}_{tag}.json"
        if rc == 0 and result.exists():
            data = json.loads(result.read_text())
            data.update(frozen_speedup(data))
            data.pop("log", None)
            summary["runs"][f"{run}_{tag}"] = data
        else:
            summary["runs"][f"{run}_{tag}"] = {"error": f"exit code {rc}"}
        out_path.write_text(json.dumps(summary, indent=2))
    print(f"-> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
