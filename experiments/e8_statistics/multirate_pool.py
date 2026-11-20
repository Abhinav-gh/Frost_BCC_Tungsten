"""E8: pool independent seeds per sampler and compare the pools, with honest errors.

One 50 ps arm cannot tell a student bias of 0.1 meV/atom from noise. The gate
reads the student alone as admissible at +1.4 se, and a single Frost seed's S_stat
moves from 2.19 to 3.57 on its tau estimate alone (plan section 1.15). This pools
independent seeds of each sampler and compares the pools.

Errors are the honest ones. Within a group the mean is weighted by each run's own
standard error, and the result is inflated by the Birge ratio sqrt(chi^2/dof)
whenever the seeds disagree by more than those errors allow: a glass that settles
differently per seed carries scatter no single run's block average can see. A
difference between groups is read in units of the combined, inflated errors.

Groups are given as name=tag1,tag2,... and the first is the reference, normally
the teacher at the same step as everything else so that the integrator's own bias
cancels. S_stat for each group against the reference combines two measured
ratios. Cost is milliseconds per simulated femtosecond, each run divided by the
teacher rate its own probe measured on the same load. Sampling efficiency is the
pooled integrated autocorrelation time in femtoseconds.

Usage
  multirate_pool.py --group teacher4=s87_ang_dt4,s87_ang_dt4_s1,f6e_mr_teacher_dt4_s2 \\
                    --group student4=f6e_mr_student_dt4,f6e_mr_student_dt4_s1 ...
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

# run as a script, so its own directory is first on sys.path
from multirate_gate import (
    OUT_DIR,
    ROOT,
    SILICA_ANGLE,
    block_se,
    frame_angles,
    load_arm,
    parse_angle,
)


def pooled(values: list[float], errors: list[float]) -> tuple[float, float, float]:
    """Inverse-variance mean, its error inflated by the Birge ratio, and that ratio."""
    x, s = np.asarray(values, dtype=float), np.asarray(errors, dtype=float)
    w = 1.0 / s ** 2
    mean = float((w * x).sum() / w.sum())
    se = float(1.0 / math.sqrt(w.sum()))
    if len(x) < 2:
        return mean, se, 1.0
    birge = math.sqrt(float((w * (x - mean) ** 2).sum()) / (len(x) - 1))
    return mean, se * max(1.0, birge), birge


def main() -> int:
    import h5py

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="silica1152_T300")
    ap.add_argument("--group", action="append", required=True, help="name=tag,tag,...; first is the reference")
    ap.add_argument("--out", default=None)
    ap.add_argument("--angle", default=SILICA_ANGLE, help="centre_Z:neighbour_Z:cutoff_A, as multirate_gate.py")
    ap.add_argument("--angle-stat", choices=("mean", "sd"), default="mean", help="as multirate_gate.py")
    args = ap.parse_args()

    with h5py.File(ROOT / "data" / "trajectories" / f"{args.run}.h5", "r") as fh:
        numbers = np.asarray(fh.attrs["numbers"], dtype=np.int64)
        cell = np.asarray(fh.attrs["cell"], dtype=np.float64)
    centres, neighbours, cutoff = parse_angle(args.angle, numbers)
    lengths = np.diag(cell)

    def angle_mean(tag: str, burn: float) -> tuple[float, float]:
        frames = np.load(OUT_DIR / f"{tag}.npz")["frames"].astype(np.float64)
        frames = frames[int(burn * len(frames)):]
        stat = np.mean if args.angle_stat == "mean" else np.std
        per_frame = np.array([stat(frame_angles(f, neighbours, centres, lengths, cutoff)) for f in frames])
        return float(per_frame.mean()), block_se(per_frame)

    groups = []
    for spec in args.group:
        name, tags = spec.split("=", 1)
        runs = []
        for tag in tags.split(","):
            arm = load_arm(tag, args.run)
            arm["angle"], arm["angle_se"] = angle_mean(tag, arm["frame_burn"])
            # teacher steps per second on this run's own load; priced below at the
            # reference group's step, which is the baseline's admissible step
            arm["teacher_rate"] = (0.5 * (arm["probe"]["before"] + arm["probe"]["after"])
                                   if "probe" in arm else float("nan"))
            runs.append(arm)
        e, e_se, e_birge = pooled([r["epot"] for r in runs], [r["epot_se"] for r in runs])
        a, a_se, a_birge = pooled([r["angle"] for r in runs], [r["angle_se"] for r in runs])
        # tau's own error is Madras-Sokal's: tau_se / tau = ess_per_second_se / ess_per_second
        taus = [r["tau_fs"] for r in runs]
        tau_ses = [r["tau_fs"] * r["ess_per_second_se"] / r["ess_per_second"] for r in runs]
        t, t_se, t_birge = pooled(taus, tau_ses)
        # tau of the per-step scalar, where every run in the group recorded one
        if all("scalar_tau_fs" in r for r in runs):
            st, st_se, _ = pooled([r["scalar_tau_fs"] for r in runs], [r["scalar_tau_se_fs"] for r in runs])
        else:
            st = st_se = float("nan")
        groups.append({"runs": runs, "name": name, "tags": [r["tag"] for r in runs], "n": len(runs),
                       "epot": e, "epot_se": e_se, "epot_birge": e_birge,
                       "angle": a, "angle_se": a_se, "angle_birge": a_birge,
                       "tau_fs": t, "tau_fs_se": t_se, "tau_birge": t_birge,
                       "scalar_tau_fs": st, "scalar_tau_fs_se": st_se,
                       "ms_per_fs": float(np.mean([r["ms_per_fs"] for r in runs])),
                       "per_run": [{"tag": r["tag"], "epot": r["epot"], "epot_se": r["epot_se"],
                                    "angle": r["angle"], "angle_se": r["angle_se"], "tau_fs": r["tau_fs"],
                                    "ms_per_fs": r["ms_per_fs"]} for r in runs]})

    ref = groups[0]
    steps = {r["config"].get("dt") or r["config"].get("inner_dt") for r in ref["runs"]}
    if len(steps) != 1:
        raise SystemExit(f"the reference group mixes teacher steps {sorted(steps)}")
    ref_dt = steps.pop()
    for g in groups:
        # the baseline's milliseconds per simulated fs on each run's own load, over the run's own
        costs = np.array([(1000.0 / (ref_dt * r["teacher_rate"])) / r["ms_per_fs"] for r in g.pop("runs")])
        g["cost_ratio"] = float(np.nanmean(costs)) if np.isfinite(costs).any() else float("nan")
    print(f"reference {ref['name']}: teacher at {ref_dt} fs; cost ratios priced at that step")
    print(f"{'group':14s} {'n':>2s} {'Epot - ref meV/atom':>22s} {'z':>6s} {'angle - ref deg':>18s} "
          f"{'z':>6s} {'tau fs':>12s} {'cost':>6s} {'S_stat':>12s}")
    for g in groups:
        de = 1e3 * (g["epot"] - ref["epot"])
        de_se = 1e3 * math.hypot(g["epot_se"], ref["epot_se"]) if g is not ref else 1e3 * g["epot_se"]
        da = g["angle"] - ref["angle"]
        da_se = math.hypot(g["angle_se"], ref["angle_se"]) if g is not ref else g["angle_se"]
        g["vs_ref"] = {"epot_meV": de, "epot_se_meV": de_se, "epot_z": de / de_se if g is not ref else 0.0,
                       "angle_deg": da, "angle_se_deg": da_se, "angle_z": da / da_se if g is not ref else 0.0}
        s_text = "-"
        if g is not ref and g["cost_ratio"] == g["cost_ratio"]:
            tau_ratio = ref["tau_fs"] / g["tau_fs"]
            tau_rel = math.hypot(ref["tau_fs_se"] / ref["tau_fs"], g["tau_fs_se"] / g["tau_fs"])
            g["S_stat"] = g["cost_ratio"] * tau_ratio
            g["S_stat_se"] = g["S_stat"] * tau_rel
            s_text = f"{g['S_stat']:.2f}+/-{g['S_stat_se']:.2f}"
            if ref["scalar_tau_fs"] == ref["scalar_tau_fs"] and g["scalar_tau_fs"] == g["scalar_tau_fs"]:
                # priced on the per-step scalar instead: one cadence for both arms. A different
                # observable, so it is reported beside the energy-priced number, never merged
                s_ratio = ref["scalar_tau_fs"] / g["scalar_tau_fs"]
                s_rel = math.hypot(ref["scalar_tau_fs_se"] / ref["scalar_tau_fs"],
                                   g["scalar_tau_fs_se"] / g["scalar_tau_fs"])
                g["S_stat_scalar"] = g["cost_ratio"] * s_ratio
                g["S_stat_scalar_se"] = g["S_stat_scalar"] * s_rel
                s_text += f" | scalar {g['S_stat_scalar']:.2f}+/-{g['S_stat_scalar_se']:.2f}"
        v = g["vs_ref"]
        print(f"{g['name']:14s} {g['n']:2d} {v['epot_meV']:+12.3f} +/- {v['epot_se_meV']:.3f} {v['epot_z']:+6.1f} "
              f"{v['angle_deg']:+9.3f} +/- {v['angle_se_deg']:.3f} {v['angle_z']:+6.1f} "
              f"{g['tau_fs']:6.0f}+/-{g['tau_fs_se']:<4.0f} {g['cost_ratio']:6.2f} {s_text:>12s}"
              f"   (Birge E {g['epot_birge']:.2f}, angle {g['angle_birge']:.2f}, tau {g['tau_birge']:.2f};"
              f" scalar tau {g['scalar_tau_fs']:.0f} fs)",
              flush=True)

    out = OUT_DIR / (args.out or f"multirate_pool_{args.run}.json")
    out.write_text(json.dumps({"run": args.run, "reference": ref["name"], "groups": groups},
                              indent=2, default=float))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
