"""Kanda et al.'s accuracy test on our runs of their released code (gaps.md, Part D, Kanda T1 pre-registration).

Their criterion: the 1:3 switching run's partial RDFs (Ti-Ti, Ti-O, O-O) correlate with the 6 A model's at Pearson
r >= 0.995. Their script (`third_party/deepmd-mts/examples/scale_mts_tio2/analysis/plot_rdf_pair_compare.py`) averages
OVITO's partial RDF (CoordinationAnalysisModifier, cutoff 8 A, 200 bins) over a dump's frames; OVITO is not installed
here, so this computes the same partial RDFs with numpy and a periodic k-d tree (orthogonal NPT boxes, read per
frame). Pearson r does not depend on the normalisation; it is reported over every bin (as their table of g(r) would
give) and over the bins where either curve is nonzero.

Usage: python experiments/e14_baselines/kanda_rdf.py REF_DIR RUN_DIR [RUN_DIR ...] [--skip-ps 10]
CPU only (LANE=check).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

CUTOFF, BINS = 8.0, 200
TYPES = {1: "Ti", 2: "O"}
PAIRS = ((1, 1), (1, 2), (2, 2))


def frames(path: Path):
    """Yield (timestep, box lengths, types, positions) from a LAMMPS custom dump with xu yu zu."""
    with path.open() as fh:
        while True:
            line = fh.readline()
            if not line:
                return
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            step = int(fh.readline())
            fh.readline()
            n = int(fh.readline())
            head = fh.readline()
            bounds = [list(map(float, fh.readline().split())) for _ in range(3)]
            if "xy" in head and any(abs(b[2]) > 1e-9 for b in bounds):
                raise ValueError("triclinic box: this reader handles orthogonal boxes only")
            lo = np.array([b[0] for b in bounds])
            box = np.array([b[1] - b[0] for b in bounds])
            cols = fh.readline().split()[2:]
            data = np.loadtxt([fh.readline() for _ in range(n)])
            t = data[:, cols.index("type")].astype(int)
            x = data[:, [cols.index("xu"), cols.index("yu"), cols.index("zu")]]
            yield step, box, t, np.mod(x - lo, box)


def partial_rdf(box, types, x):
    edges = np.linspace(0.0, CUTOFF, BINS + 1)
    shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    tree = cKDTree(x, boxsize=box)
    pairs = tree.query_pairs(CUTOFF, output_type="ndarray")
    d = x[pairs[:, 1]] - x[pairs[:, 0]]
    d -= box * np.round(d / box)
    r = np.linalg.norm(d, axis=1)
    ti, tj = types[pairs[:, 0]], types[pairs[:, 1]]
    vol = float(np.prod(box))
    out = {}
    for a, b in PAIRS:
        sel = ((ti == a) & (tj == b)) | ((ti == b) & (tj == a))
        h, _ = np.histogram(r[sel], bins=edges)
        na, nb = int((types == a).sum()), int((types == b).sum())
        norm = na * (na - 1) / 2.0 if a == b else na * nb
        out[f"{TYPES[a]}-{TYPES[b]}"] = h * vol / (norm * shell)
    return 0.5 * (edges[1:] + edges[:-1]), out


def averaged(run: Path, skip_ps: float):
    acc, n, steps = None, 0, []
    for step, box, t, x in frames(run / "tio2_anatase.dump"):
        if step * 1e-3 < skip_ps:  # their inputs: timestep 0.001 ps
            continue
        r, g = partial_rdf(box, t, x)
        acc = g if acc is None else {k: acc[k] + g[k] for k in acc}
        n += 1
        steps.append(step)
    return r, {k: v / n for k, v in acc.items()}, steps


def pearson(a, b, nonzero=False):
    if nonzero:
        keep = (a > 0) | (b > 0)
        a, b = a[keep], b[keep]
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--skip-ps", type=float, default=10.0, help="drop the NPT equilibration's first part")
    ap.add_argument("--out", default="data/results/e14/kanda_t1_rdf.json")
    a = ap.parse_args()
    _, ref, ref_steps = averaged(Path(a.ref), a.skip_ps)
    report = {"reference": a.ref, "reference_frames": len(ref_steps), "skip_ps": a.skip_ps, "criterion": 0.995, "runs": {}}
    for run in a.runs:
        _, g, steps = averaged(Path(run), a.skip_ps)
        row = {"frames": len(steps)}
        for pair in ref:
            row[pair] = {"r_all_bins": pearson(ref[pair], g[pair]), "r_nonzero_bins": pearson(ref[pair], g[pair], True)}
        row["pass"] = all(min(v["r_all_bins"], v["r_nonzero_bins"]) >= 0.995 for k, v in row.items() if "-" in k)
        report["runs"][run] = row
        print(run, {k: (round(v["r_all_bins"], 5), round(v["r_nonzero_bins"], 5)) for k, v in row.items() if "-" in k},
              "pass" if row["pass"] else "FAIL")
    Path(a.out).write_text(json.dumps(report, indent=1))
    print("->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
