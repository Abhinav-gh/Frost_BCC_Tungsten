"""How long can the MD residual be held? The hold error e(k), measured (MD half).

The MD counterpart of ``experiments/e10_cfd/decorrelation.py``, with the same
definition. Along the teacher's own trajectory (1 fs frames), hold the residual
R = F_T - F_S from frame t, use it at frame t + k with a fresh student force, and
compare with the fresh teacher force:

    e(k) = || (R(t) - R(t+k)) / m || / || F_T(t+k) / m ||,

RMS over atoms and frames, in acceleration units (the certificate's units). Also
reported: D(k) of the residual (0 = identical, 1 = unrelated), and the residual's
share of the acceleration.

Systems and students are those of the production runs. The known outcomes are set
beside the curves:
- Li3PO4 passes at 6 fs;
- silica passes at 6.0 fs (0.217 T) and fails by resonance beyond about 0.22 T;
- Si fails at 14 fs (0.178 T);
- solid Cu certifies at 24 fs;
- liquid Cu runs clean at 12 fs, and at 24 fs only with rollback.

Output: data/results/e9/residual_decorrelation.json

Usage (guarded launcher; GPU)
  scripts/safe_run.sh md_decor .venvs/frost-mace/bin/python \\
      experiments/e9_direction/residual_decorrelation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "results" / "e9" / "residual_decorrelation.json"
LAGS = (1, 2, 4, 6, 8, 12, 16, 24, 32, 48)
SYSTEMS = {
    "li3po4_T300": "li3po4_T300_s35c16e1.pt",
    "silica1152_T300": "silica1152_T300_s35c16e1.pt",
    "si512_T300": "si512_T300_s35c16e1.pt",
    "cu500_T300": "cu500_T300_s35c16e1.pt",
    "cu500_T1600": "cu500_T1600_s35c16e1liq.pt",
}
WINDOWS = (2000, 5000, 8000)  # three independent stretches of each trajectory
SPAN = max(LAGS) + 48


def main() -> int:
    from mace.calculators import mace_mp

    from frost.neighbours import VerletList
    from frost.student import GraphedStudent
    from frost.students import load_student
    from frost.teacher import frame_atoms
    from frost.teacher_fast import GraphedTeacher

    dev = "cuda"
    calc = mace_mp(model="medium", device=dev, default_dtype="float32", enable_cueq=True)
    report = {"lags_fs": LAGS, "systems": {}}
    for run, student_file in SYSTEMS.items():
        with h5py.File(ROOT / "data" / "trajectories" / f"{run}.h5", "r") as fh:
            numbers = np.asarray(fh.attrs["numbers"], dtype=np.int64)
            cell = np.asarray(fh.attrs["cell"], dtype=np.float64)
            frames = {w: np.asarray(fh["positions"][w:w + SPAN], dtype=np.float64) for w in WINDOWS}
        atoms = frame_atoms(numbers, frames[WINDOWS[0]][0], cell)
        masses = torch.as_tensor(atoms.get_masses(), dtype=torch.float64, device=dev).unsqueeze(1)
        teacher = GraphedTeacher(calc, atoms, dev)
        tl = VerletList(cell, teacher.r_max, 1.0, dev, wrap_positions=False, builder="alchemi")
        student = load_student(ROOT / "data" / "models" / "students" / student_file, dev)
        if student.config.get("backend", "e3nn") != "cueq":
            student = student.converted("cueq")
        types = student.types(torch.as_tensor(numbers, device=dev))
        sl = VerletList(cell, float(student.config["cutoff"]), 0.3, dev, wrap_positions=False,
                        builder="alchemi")
        x0 = torch.as_tensor(frames[WINDOWS[0]][0], dtype=torch.float64, device=dev)
        tl.update(x0, force=True)
        teacher.set_edges(tl.i, tl.j, tl.shifts, tl.unit_shifts)
        sl.update(x0, force=True)
        graphed = GraphedStudent(student, types, len(atoms), capacity=int(1.3 * sl.i.numel()) + 64)
        graphed.set_edges(sl.i, sl.j, sl.shifts)
        a_t, a_r = [], []  # per window: (frames, atoms, 3) accelerations
        for w in WINDOWS:
            ft, fr = [], []
            for x_np in frames[w]:
                x = torch.as_tensor(x_np, dtype=torch.float64, device=dev)
                if tl.update(x)[1]:
                    teacher.set_edges(tl.i, tl.j, tl.shifts, tl.unit_shifts)
                if sl.update(x)[1]:
                    graphed.set_edges(sl.i, sl.j, sl.shifts)
                f_t = teacher(x)[1].double()
                f_s = graphed(x)[1].detach().double()
                ft.append((f_t / masses).cpu().numpy())
                fr.append(((f_t - f_s) / masses).cpu().numpy())
            a_t.append(np.stack(ft))
            a_r.append(np.stack(fr))
        e, d = {}, {}
        total = np.sqrt(np.mean([np.mean(np.sum(a**2, -1)) for a in a_t]))
        var_r = np.mean([np.mean(np.sum((r - r.mean(0)) ** 2, -1)) for r in a_r])
        for k in LAGS:
            diffs = [np.mean(np.sum((r[k:] - r[:-k]) ** 2, -1)) for r in a_r]
            e[k] = float(np.sqrt(np.mean(diffs)) / total)
            d[k] = float(np.mean(diffs) / (2 * var_r))
        share = float(np.sqrt(np.mean([np.mean(np.sum(r**2, -1)) for r in a_r])) / total)
        report["systems"][run] = {"student": student_file, "e": e, "D_residual": d,
                                  "residual_share_of_acceleration": share}
        print(f"== {run} ({student_file}): residual share of the acceleration {share:.3f}", flush=True)
        print("  e(k)  " + " ".join(f"{k}fs:{e[k]:.4f}" for k in LAGS), flush=True)
        print("  D(k)  " + " ".join(f"{k}fs:{d[k]:.3f}" for k in LAGS), flush=True)
        del teacher, graphed
        torch.cuda.empty_cache()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
