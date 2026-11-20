"""E8: the fair statistics comparison — each sampler at its own best tuning.

S_stat is a ratio of effective samples per wall second, and both arms have a knob
that changes it by more than the kernels do:

- the teacher's Langevin friction sets how fast its potential energy decorrelates
  (silica: 0.076 ESS/s at 0.002/fs, 0.393 at 0.05, 0.099 at 0.2);
- the sampler's momentum refresh does the same for the chain.

Comparing a tuned sampler against an untuned baseline measures the tuning, not the
method. This reads the runs recorded by ``sample.py``, picks the best tuning on
each side by ESS/s, and reports the ratio with both errors propagated. Runs that
share one stored teacher reference share its error, which averaging over seeds
cannot reduce, so the reference's relative error is added once at the end.

Output: data/results/e8/comparison_<run>.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "results" / "e8"


def load(run: str) -> tuple[list[dict], list[dict]]:
    """Teacher-only arms and nested arms recorded for ``run``."""
    teacher, nested = [], []
    for path in sorted(OUT_DIR.glob(f"sample_{run}*.json")):
        report = json.loads(path.read_text())
        config = report.get("config", {})
        arm = report.get("teacher" if report.get("arms") == "teacher" else "nested", {})
        if "ess_per_second_se" not in arm:
            continue  # recorded before tau_int carried an error bar (plan section 1.14)
        entry = {"tag": config.get("tag") or path.stem, "path": path.name}
        if report.get("arms") == "teacher":
            entry |= {"friction": config.get("friction"), "dt": config.get("dt"),
                      "ess_per_second": arm["ess_per_second"], "se": arm["ess_per_second_se"],
                      "tau": arm["tau_int_samples"], "epot": arm["epot_mean_eV_per_atom"],
                      "epot_se": arm["epot_mean_se"], "rdf": arm["rdf"]["g_total"],
                      "own_halves": arm.get("rdf_halves_l1", float("nan"))}
            teacher.append(entry)
        elif "nested" in report:
            entry |= {"refresh": config.get("refresh"), "extra_chances": config.get("extra_chances"),
                      "jitter": config.get("jitter"), "n_draft": config.get("n_draft"),
                      "draft_dt": config.get("draft_dt") or config.get("dt"),
                      "student": Path(config.get("student", "")).stem,
                      "acceptance": report.get("acceptance"), "tau": arm["tau_int_samples"],
                      "ess_per_second": arm["ess_per_second"], "se": arm["ess_per_second_se"],
                      "epot_difference_in_se": report["exactness"]["epot_mean_difference_in_se"]}
            nested.append(entry)
    return teacher, nested


def admissible_baseline(teacher: list[dict]) -> dict:
    """The fastest teacher arm that still samples what the reference step samples.

    Two knobs behave differently and must not be gated the same way. **Friction**
    is free: Langevin samples the same distribution at any friction, so the best
    friction at a given step is simply the one with the highest ESS/s, which is how
    the baseline was tuned in section 1.14. **The time step** is not free: ESS/s
    rises with it almost by construction -- the same cost per step covers more
    simulated time, so tau in samples falls -- while BAOAB's configurational bias
    grows as dt^2. Maximising ESS/s across steps would therefore pick a baseline
    for being fast and make S_stat fall for an artefact rather than a finding
    (frost-87).

    So: tune friction within each step, then admit a larger step only if it still
    samples the same distribution as the smallest one -- mean potential energy
    within twice the combined standard error, and an RDF within the reference's own
    half-to-half spread. The fastest survivor is the fair baseline. With a single
    step present this reduces to the old behaviour.
    """
    graded = [e for e in teacher if e.get("dt") is not None and e.get("rdf")]
    if not graded:
        return max(teacher, key=lambda e: e["ess_per_second"])

    by_step: dict[float, list[dict]] = {}
    for entry in graded:
        by_step.setdefault(entry["dt"], []).append(entry)
    def usable(arms: list[dict]) -> list[dict]:
        # an arm without its own half-to-half spread cannot serve as the noise
        # reference; arms recorded before that field exists still compete on speed
        return [a for a in arms if a.get("own_halves") == a.get("own_halves")] or arms

    best_at = {dt: max(arms, key=lambda e: e["ess_per_second"]) for dt, arms in by_step.items()}
    reference_at = {dt: max(usable(arms), key=lambda e: e["ess_per_second"])
                    for dt, arms in by_step.items()}

    step = min(reference_at)
    reference = reference_at[step]
    # friction is free at a fixed step, so the fastest arm at the reference's own
    # step is admissible by construction and is not bias-tested against it
    survivors_at_reference_step = best_at[step]
    if len(best_at) == 1:
        return survivors_at_reference_step

    survivors = [survivors_at_reference_step]
    for dt, candidate in sorted(best_at.items()):
        if dt == step:
            continue
        g_ref = np.asarray(reference["rdf"], dtype=float)
        g_can = np.asarray(candidate["rdf"], dtype=float)
        candidate["bias_rdf_l1"] = float(np.abs(g_can - g_ref).sum() / np.abs(g_ref).sum())
        # Each arm's own half-to-half L1 is measured on half-length samples, which
        # are noisier by sqrt(2), so it implies a full-run noise of h/sqrt(2) and two
        # independent arms differ by sqrt((h_ref^2 + h_cand^2)/2). Judging against the
        # reference's raw halves instead makes the gate a function of how long the
        # reference happened to run: silica's reference reads 0.0028 at 2400 s and
        # 0.005-0.009 at 1200 s, which would reject or admit a 4 fs arm on sampling
        # effort alone.
        noise = math.sqrt((reference["own_halves"] ** 2 + candidate["own_halves"] ** 2) / 2)
        candidate["bias_noise_floor"] = noise
        combined = math.hypot(candidate["epot_se"], reference["epot_se"])
        gap = abs(candidate["epot"] - reference["epot"])
        candidate["bias_epot_se"] = gap / combined if combined else float("inf")
        unbiased = candidate["bias_rdf_l1"] <= noise and gap <= 2.0 * combined
        candidate["admissible"] = unbiased
        if unbiased:
            survivors.append(candidate)
    return max(survivors, key=lambda e: e["ess_per_second"])


def pooled(entries: list[dict]) -> tuple[float, float]:
    """Mean ESS/s of repeated runs and the standard error of that mean.

    Seeds of one configuration are independent chains, so the spread between them
    is the honest error; with a single run its own Sokal error stands in.
    """
    values = [e["ess_per_second"] for e in entries]
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, entries[0]["se"]
    spread = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
    return mean, spread / math.sqrt(len(values))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="silica1152_T300")
    ap.add_argument("--kernel", default="extra_chances=2,jitter=0.5",
                    help="which nested runs to pool for the headline, as key=value pairs")
    args = ap.parse_args()

    teacher, nested = load(args.run)
    if not teacher:
        print(f"no teacher-only arms recorded for {args.run}; run sample.py --arms teacher")
        return 1
    best_t = admissible_baseline(teacher)

    wanted = dict(pair.split("=") for pair in args.kernel.split(",")) if args.kernel else {}
    def matches(entry: dict) -> bool:
        return all(str(entry.get(key)) == value for key, value in wanted.items())

    # everything that changes the kernel is part of the key: pooling seeds of
    # different students averages a 0.70-acceptance run with a 0.14 one, and pooling
    # draft lengths averages chains whose tau differs by 2x
    groups: dict[tuple, list[dict]] = {}
    for entry in nested:
        if matches(entry):
            key = (entry["student"], entry["draft_dt"], entry["n_draft"], entry["refresh"])
            groups.setdefault(key, []).append(entry)

    print(f"{args.run}: teacher arms (ESS/s; the fastest step that is still unbiased)")
    for e in sorted(teacher, key=lambda e: (e.get("dt") or 0, e["friction"] or 0)):
        mark = " <- best" if e is best_t else ""
        bias = ""
        if "bias_rdf_l1" in e:
            bias = (f"  bias: RDF {e['bias_rdf_l1']:.4f} vs noise "
                    f"{e.get('bias_noise_floor', float('nan')):.4f}, epot {e['bias_epot_se']:+.1f} se")
        print(f"  dt {e.get('dt', '?')} friction {e['friction']:<7} {e['ess_per_second']:.3f}"
              f" +/- {e['se']:.3f}  tau {e['tau']:.0f}{bias}{mark}")

    rows = []
    print(f"\nnested arms matching {args.kernel or 'any'} (pooled over seeds, per student)")
    for (student, dt, n_draft, refresh), entries in sorted(groups.items(),
                                                           key=lambda kv: tuple(x or 0 for x in kv[0][1:])):
        mean, se = pooled(entries)
        ratio = mean / best_t["ess_per_second"]
        rel = math.hypot(se / mean, best_t["se"] / best_t["ess_per_second"])
        rows.append({"student": student, "draft_dt": dt, "n_draft": n_draft,
                     "refresh": refresh, "runs": len(entries), "ess_per_second": mean,
                     "se": se, "S_stat_fair": ratio, "S_stat_fair_se": ratio * rel,
                     "acceptance": sum(e["acceptance"] for e in entries) / len(entries),
                     "tags": [e["tag"] for e in entries]})
        print(f"  {student:<26} {dt} fs x{n_draft:<3} refresh {refresh:<5} n={len(entries)}"
              f"  {mean:.3f} +/- {se:.3f}"
              f"  acceptance {rows[-1]['acceptance']:.2f}"
              f"  S_stat(fair) {ratio:.2f} +/- {ratio * rel:.2f}")

    report = {"run": args.run, "kernel": args.kernel, "teacher_best": best_t,
              "teacher_arms": teacher, "nested_by_student_and_refresh": rows}
    out = OUT_DIR / f"comparison_{args.run}.json"
    out.write_text(json.dumps(report, indent=2))
    if rows:
        best = max(rows, key=lambda r: r["S_stat_fair"])
        print(f"\nfair headline: S_stat = {best['S_stat_fair']:.2f} +/- {best['S_stat_fair_se']:.2f}x "
              f"({best['student']}, {best['draft_dt']} fs x {best['n_draft']}, refresh "
              f"{best['refresh']}, against the teacher's best friction "
              f"{best_t['friction']}/fs) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
