"""Where a water residual's staleness lives: intramolecular coordinates against rigid-body motion (e2e for gap water).

The certificate refuses water at k = 2 on the residual's staleness (3.5 against 2.0, in acceleration units), and its
residual decorrelates within a few fs (hold staleness at 2 fs is a third of its level; linear extrapolation is worse
than holding). This measures why, on unseen windows of the teacher's own trajectory:

1. R = F_teacher - F_student on consecutive frames (the trajectory's 1 fs steps).
2. Per molecule (O, H, H in file order), the internal coordinates q = (r_OH1, r_OH2, theta_HOH) and their Wilson B
   matrix (dq/dx, 3 x 9). R splits exactly into an intramolecular part B^T (B B^T)^-1 B R (the generalised forces
   Q on q, mapped back) and a rigid-body part orthogonal to it (B annihilates translations and rotations).
3. The staleness of each part at lag k in acceleration units, per species, as the certificate measures it
   (rms per component of dR / m, with the run's repartitioned masses, ``--hmr``).
4. How much of the intramolecular change is linear in the coordinates' own change: per lag, the pooled fit
   dQ = K dq (K a 3 x 3 matrix shared by all molecules: stretch, bend and their coupling) and the staleness left
   after subtracting it. If K explains most of it, the residual's fast part is a stiffness mismatch that an online
   internal-coordinate model can track every inner step, leaving only the slow rigid-body residual to the impulse.

Usage (guarded launcher; GPU)
  LANE=check scripts/safe_run.sh residual_modes env PYTHONPATH=src .venvs/frost-mace/bin/python \\
      experiments/e5_student/residual_modes.py --run water216off_T300 \\
      --student data/models/students/water216off_T300_f45c16.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from frost.neighbours import edges_for
from frost.student import energy_and_forces
from frost.students import load_student
from frost.teacher import frame_atoms, mace_forces
from frost.teachers import load_calc, teacher_of
from frost.trajectory import TrajectoryReader

ROOT = Path(__file__).resolve().parents[2]
TRAJ = ROOT / "data" / "trajectories"
OUT = ROOT / "data" / "results" / "e5"
AMU_ACC = 1e3  # forces in eV/A, masses in amu: report accelerations in meV/(A amu), the certificate's units


def mic(d: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Minimum-image vectors for an orthorhombic or general cell (fractional rounding)."""
    inv = np.linalg.inv(cell)
    frac = d @ inv
    return (frac - np.round(frac)) @ cell


def wilson_b(x: np.ndarray, cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per molecule (O, H1, H2 in order): q (m, 3) = (r1, r2, theta) and B (m, 3, 9), dq / d(xO, xH1, xH2)."""
    o, h1, h2 = x[0::3], x[1::3], x[2::3]
    d1, d2 = mic(h1 - o, cell), mic(h2 - o, cell)
    r1, r2 = np.linalg.norm(d1, axis=1), np.linalg.norm(d2, axis=1)
    u1, u2 = d1 / r1[:, None], d2 / r2[:, None]
    cos = np.clip((u1 * u2).sum(1), -1.0, 1.0)
    sin = np.sqrt(1.0 - cos**2)
    theta = np.arccos(cos)
    m = len(o)
    b = np.zeros((m, 3, 9))
    b[:, 0, 0:3], b[:, 0, 3:6] = -u1, u1
    b[:, 1, 0:3], b[:, 1, 6:9] = -u2, u2
    gh1 = (cos[:, None] * u1 - u2) / (r1 * sin)[:, None]
    gh2 = (cos[:, None] * u2 - u1) / (r2 * sin)[:, None]
    b[:, 2, 3:6], b[:, 2, 6:9], b[:, 2, 0:3] = gh1, gh2, -(gh1 + gh2)
    return np.stack([r1, r2, theta], axis=1), b


def split(r: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(generalised forces Q (m, 3), intramolecular part (n, 3), rigid-body part (n, 3)) of a force r (n, 3)."""
    m = b.shape[0]
    f = r.reshape(m, 9)
    q = np.linalg.solve(b @ b.transpose(0, 2, 1), np.einsum("mij,mj->mi", b, f))
    intra = np.einsum("mji,mj->mi", b, q)
    return q, intra.reshape(-1, 3), (f - intra).reshape(-1, 3)


def rms_acc(d: np.ndarray, masses: np.ndarray, sel=None) -> float:
    a = d / masses[:, None]
    if sel is not None:
        a = a[sel]
    return float(AMU_ACC * np.sqrt((a**2).sum(1).mean() / 3.0))  # per component, as the runtime's staleness


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="water216off_T300")
    ap.add_argument("--student", required=True)
    ap.add_argument("--starts", default="4000,7000,9000", help="unseen windows (replay's)")
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--lags", default="1,2,4,8")
    ap.add_argument("--hmr", type=float, default=3.0, help="Da moved to each H, as the runtime runs (0: real masses)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    dev = args.device
    lags = [int(k) for k in args.lags.split(",")]

    path = TRAJ / f"{args.run}.h5"
    with TrajectoryReader(path) as reader:
        numbers, cell = reader.numbers.copy(), reader.cell.copy()
        teacher_name = teacher_of(reader.attrs)
        windows = [[np.asarray(reader.positions[s + j], dtype=np.float64) for j in range(args.window + 1)]
                   for s in (int(s) for s in args.starts.split(","))]
    assert (numbers[0::3] == 8).all() and (numbers[1::3] == 1).all() and (numbers[2::3] == 1).all(), "O, H, H order"
    from ase.data import atomic_masses

    from frost.teachers import repartition_hydrogen

    masses = atomic_masses[numbers]
    if args.hmr:
        masses = np.asarray(repartition_hydrogen(numbers, masses, args.hmr), dtype=np.float64)
    is_h = numbers == 1
    teacher = load_calc(teacher_name, dev, "float32", enable_cueq=True)
    student = load_student(args.student, dev)
    types = student.types(torch.as_tensor(numbers, device=dev))
    cutoff = float(student.config["cutoff"])

    rows = {k: {"all": [], "intra": [], "rigid": [], "intra_after_K": [], "H_all": [], "O_all": [],
                "H_intra": [], "H_rigid": [], "O_rigid": []} for k in lags}
    fits = {k: {"num": np.zeros((3, 3)), "den": np.zeros((3, 3)), "pairs": []} for k in lags}
    levels = []
    for window in windows:
        res, qs, gq, intra, rigid = [], [], [], [], []
        for x in window:
            atoms = frame_atoms(numbers, x, cell)
            f_t, _, _ = mace_forces(teacher, atoms, dev)
            xw, ei, ej, sh, _ = edges_for(x, cell, cutoff, dev)
            _, f_s = energy_and_forces(student, xw, types, ei, ej, sh)
            r = np.asarray(f_t, dtype=np.float64) - f_s.detach().double().cpu().numpy()
            q, b = wilson_b(x, cell)
            g, ri, rr = split(r, b)
            res.append(r), qs.append(q), gq.append(g), intra.append(ri), rigid.append(rr)
            levels.append(rms_acc(r, masses))
        for k in lags:
            for t in range(len(window) - k):
                d_all, d_in, d_rb = res[t + k] - res[t], intra[t + k] - intra[t], rigid[t + k] - rigid[t]
                rows[k]["all"].append(rms_acc(d_all, masses))
                rows[k]["intra"].append(rms_acc(d_in, masses))
                rows[k]["rigid"].append(rms_acc(d_rb, masses))
                rows[k]["H_all"].append(rms_acc(d_all, masses, is_h))
                rows[k]["O_all"].append(rms_acc(d_all, masses, ~is_h))
                rows[k]["H_intra"].append(rms_acc(d_in, masses, is_h))
                rows[k]["H_rigid"].append(rms_acc(d_rb, masses, is_h))
                rows[k]["O_rigid"].append(rms_acc(d_rb, masses, ~is_h))
                dq, dg = qs[t + k] - qs[t], gq[t + k] - gq[t]
                fits[k]["num"] += dg.T @ dq
                fits[k]["den"] += dq.T @ dq
                fits[k]["pairs"].append((t, dq, dg, window, res, qs))
    report = {"run": args.run, "student": args.student, "teacher": teacher_name, "window": args.window, "hmr": args.hmr,
              "starts": args.starts, "level_acc": float(np.mean(levels)), "lags": {}}
    print(f"{args.run} with {Path(args.student).name}: residual level {report['level_acc']:.2f} meV/(A amu) "
          f"(certificate bound 2.0)", flush=True)
    for k in lags:
        kmat = np.linalg.solve(fits[k]["den"].T, fits[k]["num"].T).T  # dQ ~ K dq, least squares pooled
        after, explained = [], []
        for t, dq, dg, window, res, _qs in fits[k]["pairs"]:
            pred = dq @ kmat.T
            explained.append(1.0 - ((dg - pred) ** 2).sum() / max((dg**2).sum(), 1e-30))
            _, b = wilson_b(window[t + k], cell)
            corr = np.einsum("mji,mj->mi", b, pred).reshape(-1, 3)
            after.append(rms_acc(res[t + k] - res[t] - corr, masses))
        row = {name: float(np.mean(v)) for name, v in rows[k].items() if v}
        row["intra_after_K"] = float(np.mean(after))
        row["K_r2_on_Q"] = float(np.mean(explained))
        row["K"] = kmat.tolist()
        report["lags"][str(k)] = row
        print(f"  k {k:2d}: staleness {row['all']:.2f} = intra {row['intra']:.2f} (+) rigid {row['rigid']:.2f} | "
              f"H {row['H_all']:.2f} (intra {row['H_intra']:.2f}, rigid {row['H_rigid']:.2f}), O {row['O_all']:.2f} "
              f"(rigid {row['O_rigid']:.2f}) | linear K explains {100 * row['K_r2_on_Q']:.0f}% of dQ, "
              f"total left {row['intra_after_K']:.2f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out = OUT / f"residual_modes_{args.run}_{Path(args.student).stem}{tag}.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
