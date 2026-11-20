"""E8: gate Frost's multirate samplers against the teacher, and price the survivors.

Reads arms written by ``multirate_sample.py`` (candidates) and by ``sample.py
--arms teacher`` (the reference and the baseline), and applies to every candidate
the gates that admitted the teacher's 4 fs step (plan section 1.15):

  energy     mean teacher potential energy per atom, within 2 combined standard
             errors of the reference. The most sensitive gate: it rejects the
             teacher's own 8 fs step at +18.6 se, which the angles pass (frost-87);
  variance   its fluctuations: the ratio to the reference within 2 standard errors,
             where a variance estimated from ESS effective samples has a relative
             error of sqrt(2/ESS). 8 fs reads 1.22, admissible 4 fs 1.03-1.09;
  RDF        L1 against the reference, within the null range: the largest L1
             between any two admissible teacher arms given (reference, baseline
             seeds, --null). The half-to-half floor sqrt((h_ref^2 + h_cand^2)/2)
             is not a bound. On silica, admissible arms differ by 0.0023-0.0135,
             by seed more than by step, against floors of 0.0075-0.0111, so that
             floor rejects about half of all unbiased arms. It rejected a 4 fs seed
             that every other gate admits, while dt = 8 fs, 18.6 se off in energy,
             reads 0.0160: the RDF is the weakest of the four;
  angles     the mean bond angle at a chosen centre (--angle; Si-O-Si at oxygen
             for silica, O-P-O at phosphorus for Li3PO4), within 2 combined block
             standard errors. Every stored frame is used, not a subsample: a shift
             is read against its own error bar, never against zero or a two-seed
             spread. Around a tetrahedral centre the mean is pinned by geometry
             (sum of cos theta_ij = (|sum u_i|^2 - 4)/2), so --angle-stat sd reads
             the per-frame width there instead.

  drift      reported, not gated until calibrated: last-quarter minus
             first-quarter teacher energy against its block error. Every gate above
             compares run MEANS, and a run that wanders late (Li3PO4's student alone
             on seed 1 ends +0.30 meV/atom above its own mean) passes a mean gate
             that a longer run would fail (frost-87, importing the halves test that
             50 ps NVE drift needed). The teacher's own arms set the null.

A candidate survives only if it passes all four; none is sufficient alone. The first candidate should be
the student alone, the arm every gate must reject; a gate it passes is reported
as powerless, because it cannot tell Frost from not running the teacher.

Survivors are priced against the baseline at this run's own teacher rate: the
baseline's effective samples per teacher step (from its stored arms) times the
steps per second the candidate's run measured for the teacher on the same load.
That keeps a busy server from inflating or deflating S_stat (traps 13, 13a).

Usage
  multirate_gate.py --reference s87_ang_dt1 --baseline s87_ang_dt4 s87_ang_dt4_s1 \\
      --candidates f6e_mr_student_dt1 f6e_mr_respa_dt2_k8 ...
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "results" / "e8"
# silica's Si-O-Si angle at the bridging oxygen: centre O, neighbours Si, within the
# first minimum of the Si-O radial distribution
SILICA_ANGLE = "8:14:2.0"


def parse_angle(spec: str, numbers: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """'centre_Z:neighbour_Z:cutoff_A' -> centre indices, neighbour indices, cutoff."""
    centre, neighbour, cutoff = spec.split(":")
    return (np.flatnonzero(numbers == int(centre)), np.flatnonzero(numbers == int(neighbour)),
            float(cutoff))


def frame_angles(frame: np.ndarray, neighbours: np.ndarray, centres: np.ndarray, lengths: np.ndarray,
                 cutoff: float = 2.0) -> np.ndarray:
    """Every neighbour-centre-neighbour angle in one frame, in degrees (orthorhombic cell).

    With the default arguments of the silica gate this is the Si-O-Si angle at each
    oxygen. Vectorised over all centre-neighbour pairs; a centre with more than two
    neighbours contributes every pair, as angle_gate.py does.
    """
    d = frame[neighbours][None, :, :] - frame[centres][:, None, :]  # (n_centres, n_neighbours, 3)
    d -= lengths * np.round(d / lengths)
    r = np.linalg.norm(d, axis=2)
    # a same-species angle (Si-Si-Si, Cu-Cu-Cu) draws centres and neighbours from one
    # set, so every atom is its own neighbour at r = 0. Left in, it adds a
    # zero-length vector that makes the cosine NaN, and inflates the coordination by
    # one: si512 reads 5 neighbours instead of 4 and cu500 13 instead of 12.
    near = (r < cutoff) & (r > 1e-6)
    out = []
    counts = near.sum(axis=1)
    two = np.flatnonzero(counts == 2)
    if len(two):
        idx = np.argsort(~near[two], axis=1, kind="stable")[:, :2]
        u = d[two, idx[:, 0]]
        v = d[two, idx[:, 1]]
        cos = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
        out.append(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    for o in np.flatnonzero(counts > 2):
        nb = np.flatnonzero(near[o])
        for a in range(len(nb)):
            for b in range(a + 1, len(nb)):
                u, v = d[o, nb[a]], d[o, nb[b]]
                cos = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
                out.append(np.atleast_1d(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))))
    return np.concatenate(out) if out else np.empty(0)


def block_se(values: np.ndarray, blocks: int = 10) -> float:
    parts = np.array_split(np.asarray(values, dtype=float), blocks)
    means = np.array([p.mean() for p in parts if len(p)])
    return float(means.std(ddof=1) / math.sqrt(len(means)))


def load_arm(tag: str, run: str) -> dict:
    """One arm's energy statistics, RDF, frames and rate, from either harness."""
    multirate = OUT_DIR / f"multirate_{run}_{tag}.json"
    teacher = OUT_DIR / f"sample_{run}_{tag}.json"
    if multirate.exists():
        report = json.loads(multirate.read_text())
        arm = report["multirate"]
        cfg = report["config"]
        kept = cfg["sim_ps"] / (cfg["sim_ps"] + cfg["burn_ps"])
        entry = {"kind": report["sampler"], "config": cfg,
                 "probe": report["teacher_probe_steps_per_second"],
                 # steady state on both sides: ESS over the wall time of the kept part
                 "ess_per_second_kept": arm["ess"] / (arm["wall_seconds"] * kept),
                 "ms_per_fs": arm["ms_per_simulated_fs"], "tau_fs": arm["tau_int_fs"],
                 # multirate_sample.py keeps frames only after its burn-in
                 "frame_burn": 0.0}
        if report["sampler"] == "teacher":
            # one energy per teacher step, all after burn-in: ESS per kept step, as for sample.py
            entry["ess_per_step"] = arm["ess"] / arm["samples"]
        scalar = arm.get("angle_scalar")
        if scalar:
            # the per-step configurational scalar: tau at one cadence in every arm
            entry["scalar_tau_fs"] = scalar["tau_int_fs"]
            entry["scalar_tau_se_fs"] = scalar["tau_int_fs"] * scalar["tau_int_se"] / scalar["tau_int_samples"]
    elif teacher.exists():
        report = json.loads(teacher.read_text())
        arm = report["teacher"]
        dt = report["config"]["dt"]
        entry = {"kind": "teacher", "config": report["config"],
                 "ms_per_fs": 1000.0 * arm["wall_seconds"] / (arm["steps_or_proposals"] * dt),
                 "tau_fs": arm["tau_int_samples"] * dt,
                 # effective samples per *kept* step: sample.py's ESS counts the
                 # steps after its 10% burn-in, so dividing by every step would
                 # deflate the baseline by 10% and inflate S_stat by as much
                 "ess_per_step": arm["ess"] / arm["samples"],
                 # sample.py's frames are summarised after a 10% burn-in
                 "frame_burn": 0.1}
    else:
        raise SystemExit(f"no arm {tag}: neither {multirate.name} nor {teacher.name}")
    frames_file = OUT_DIR / f"{tag}.npz"
    series = np.load(frames_file)["energies"] if frames_file.exists() else np.empty(0)
    series = np.asarray(series, dtype=float)[int(entry["frame_burn"] * len(series)):]
    if len(series) >= 40:
        quarter = len(series) // 4
        first, last = series[:quarter], series[-quarter:]
        entry["drift_meV"] = 1e3 * float(last.mean() - first.mean())
        entry["drift_se_meV"] = 1e3 * math.hypot(block_se(first, 5), block_se(last, 5))
    else:
        entry["drift_meV"] = entry["drift_se_meV"] = float("nan")
    entry |= {"tag": tag, "epot": arm["epot_mean_eV_per_atom"], "epot_se": arm["epot_mean_se"],
              "var": arm["epot_var_per_atom"], "rdf": np.asarray(arm["rdf"]["g_total"], dtype=float),
              "halves": arm["rdf_halves_l1"], "ess": arm["ess"], "ess_per_second": arm["ess_per_second"],
              "ess_per_second_se": arm["ess_per_second_se"]}
    return entry


def main() -> int:
    import h5py

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="silica1152_T300")
    ap.add_argument("--reference", required=True, help="the 1 fs teacher arm")
    ap.add_argument("--baseline", nargs="+", required=True, help="the admissible teacher step's seeds")
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--null", nargs="*", default=[],
                    help="further admissible teacher arms, to widen the RDF null range")
    ap.add_argument("--out", default=None, help="JSON name under data/results/e8")
    ap.add_argument("--angle", default=SILICA_ANGLE,
                    help="centre_Z:neighbour_Z:cutoff_A; Li3PO4's O-P-O is 15:8:2.0")
    ap.add_argument("--angle-range", default="120:180", help="histogram range, degrees")
    ap.add_argument("--angle-stat", choices=("mean", "sd"), default="mean",
                    help="per-frame statistic; tetrahedral centres need sd, their mean is pinned")
    args = ap.parse_args()

    with h5py.File(ROOT / "data" / "trajectories" / f"{args.run}.h5", "r") as fh:
        numbers = np.asarray(fh.attrs["numbers"], dtype=np.int64)
        cell = np.asarray(fh.attrs["cell"], dtype=np.float64)
    if np.abs(cell - np.diag(np.diag(cell))).max() > 1e-9:
        raise SystemExit("the angle gate assumes an orthorhombic cell")
    centres, neighbours, cutoff = parse_angle(args.angle, numbers)
    lengths = np.diag(cell)
    lo, hi = (float(v) for v in args.angle_range.split(":"))
    edges = np.linspace(lo, hi, 61)

    def angle_stats(tag: str, burn: float) -> dict:
        frames = np.load(OUT_DIR / f"{tag}.npz")["frames"].astype(np.float64)
        frames = frames[int(burn * len(frames)):]
        per_frame, pooled = [], []
        for f in frames:
            a = frame_angles(f, neighbours, centres, lengths, cutoff)
            per_frame.append(a.mean() if args.angle_stat == "mean" else a.std())
            pooled.append(a)
        pooled = np.concatenate(pooled)
        hist, _ = np.histogram(pooled, bins=edges, density=True)
        return {"mean": float(np.mean(per_frame)), "se": block_se(np.asarray(per_frame)),
                "sd": float(pooled.std()), "hist": hist, "frames": len(frames)}

    ref = load_arm(args.reference, args.run)
    ref["angles"] = angle_stats(args.reference, ref["frame_burn"])
    base = [load_arm(tag, args.run) for tag in args.baseline]
    for b in base:
        b["angles"] = angle_stats(b["tag"], b["frame_burn"])
    ess_per_step = np.array([b["ess_per_step"] for b in base])
    admissible = [ref] + base + [load_arm(tag, args.run) for tag in args.null]
    null_pairs = [float(np.abs(a["rdf"] - b["rdf"]).sum() / np.abs(a["rdf"]).sum())
                  for i, a in enumerate(admissible) for b in admissible[i + 1:]]
    null_range = max(null_pairs) if null_pairs else 0.0
    baseline_dt = base[0]["config"].get("dt") or base[0]["config"].get("inner_dt")
    width = edges[1] - edges[0]

    def judge(c: dict) -> dict:
        combined = math.hypot(c["epot_se"], ref["epot_se"])
        z = (c["epot"] - ref["epot"]) / combined
        rdf_l1 = float(np.abs(c["rdf"] - ref["rdf"]).sum() / np.abs(ref["rdf"]).sum())
        noise = max(null_range, math.sqrt((ref["halves"] ** 2 + c["halves"] ** 2) / 2))
        a, r = c["angles"], ref["angles"]
        shift = a["mean"] - r["mean"]
        shift_z = shift / math.hypot(a["se"], r["se"])
        ratio = c["var"] / ref["var"]
        ratio_se = ratio * math.sqrt(2.0 / c["ess"] + 2.0 / ref["ess"])
        gates = {"energy": abs(z) <= 2.0, "variance": abs(ratio - 1.0) <= 2.0 * ratio_se,
                 "rdf": rdf_l1 <= noise, "angles": abs(shift_z) <= 2.0}
        return {"epot_shift_meV": 1e3 * (c["epot"] - ref["epot"]), "epot_z": z,
                "var_ratio": ratio, "var_ratio_se": ratio_se, "rdf_l1": rdf_l1, "rdf_noise": noise,
                "angle_shift_deg": shift, "angle_shift_z": shift_z,
                "angle_hist_l1": float(np.abs(a["hist"] - r["hist"]).sum() * width),
                "drift_meV": c["drift_meV"], "drift_se_meV": c["drift_se_meV"],
                "drift_z": c["drift_meV"] / c["drift_se_meV"] if c["drift_se_meV"] else float("nan"),
                "gates": gates, "admissible": all(gates.values())}

    rows = []
    header = (f"{'arm':24s} {'dE meV/at':>9s} {'z':>6s} {'var':>11s} {'RDF L1':>7s} {'noise':>6s} "
              f"{'angle':>7s} {'z':>6s} {'drift':>7s} {'z':>6s} {'ms/fs':>6s} {'tau fs':>7s} {'S_stat':>12s}  verdict")
    print(f"reference {args.reference}: Epot {ref['epot']:.6f} +/- {ref['epot_se']:.6f}, angle "
          f"{ref['angles']['mean']:.3f} +/- {ref['angles']['se']:.3f} deg over {ref['angles']['frames']} frames")
    print(f"baseline {', '.join(args.baseline)} at {baseline_dt} fs: ESS per teacher step "
          f"{ess_per_step.mean():.4f} (seeds {', '.join(f'{v:.4f}' for v in ess_per_step)})")
    print(f"RDF null range over {len(null_pairs)} admissible pairs: "
          f"{min(null_pairs, default=0):.4f}-{null_range:.4f}")
    print(header)
    for arm in base + [load_arm(tag, args.run) for tag in args.candidates]:
        if "angles" not in arm:
            arm["angles"] = angle_stats(arm["tag"], arm["frame_burn"])
        verdict = judge(arm)
        s_stat = s_se = float("nan")
        if "probe" in arm:
            # the baseline's cost on this run's load: its samples per step, times the
            # teacher steps per second measured before and after this run
            rate = 0.5 * (arm["probe"]["before"] + arm["probe"]["after"])
            base_ess_s = ess_per_step.mean() * rate
            base_rel = ess_per_step.std(ddof=1) / math.sqrt(len(ess_per_step)) / ess_per_step.mean() \
                if len(ess_per_step) > 1 else 0.0
            s_stat = arm["ess_per_second_kept"] / base_ess_s
            s_se = s_stat * math.hypot(arm["ess_per_second_se"] / arm["ess_per_second"], base_rel)
            verdict["baseline_ess_per_second_on_this_load"] = base_ess_s
        verdict |= {"tag": arm["tag"], "kind": arm["kind"], "S_stat": s_stat, "S_stat_se": s_se,
                    "ms_per_fs": arm["ms_per_fs"], "tau_fs": arm["tau_fs"],
                    "angle_mean": arm["angles"]["mean"], "angle_se": arm["angles"]["se"]}
        rows.append(verdict)
        failed = [g for g, ok in verdict["gates"].items() if not ok]
        word = "admissible" if verdict["admissible"] else "REJECTED (" + ", ".join(failed) + ")"
        if arm["kind"] == "student" and verdict["admissible"]:
            word += " -- gates POWERLESS: the student alone passes"
        s_text = f"{s_stat:.2f}+/-{s_se:.2f}" if s_stat == s_stat else "-"
        print(f"{arm['tag']:24s} {verdict['epot_shift_meV']:+9.3f} {verdict['epot_z']:+6.1f} "
              f"{verdict['var_ratio']:5.2f}+/-{verdict['var_ratio_se']:.2f} {verdict['rdf_l1']:7.4f} {verdict['rdf_noise']:6.4f} "
              f"{verdict['angle_shift_deg']:+7.3f} {verdict['angle_shift_z']:+6.1f} "
              f"{verdict['drift_meV']:+7.3f} {verdict['drift_z']:+6.1f} {arm['ms_per_fs']:6.2f} "
              f"{arm['tau_fs']:7.0f} {s_text:>12s}  {word}", flush=True)

    out = OUT_DIR / (args.out or f"multirate_gate_{args.run}.json")
    out.write_text(json.dumps({"run": args.run, "reference": args.reference, "baseline": args.baseline,
                               "null": args.null, "rdf_null_pairs": null_pairs, "rows": rows},
                              indent=2, default=float))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
