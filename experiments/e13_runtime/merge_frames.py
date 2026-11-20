"""Score arms of a ``--no-teacher`` run against another run's teacher arm from the same start (clause ablations).

An ablation variant (one certificate clause removed) needs the same paired teacher reference as the full run it is
compared with. Rerunning a 50 ps teacher arm per variant wastes hours, and an NVE teacher arm from the same start frame
is the same trajectory to GPU round-off. This writes a frames file holding run A's teacher arm and run B's other arms
(renamed ``<arm>_<suffix>``), after checking both share atoms, cell, masses, frame strides and length, so
``score_frames.py`` scores it unchanged.

Usage: python experiments/e13_runtime/merge_frames.py A.npz B.npz SUFFIX OUT.npz [--keep-teacher NAME]

--keep-teacher NAME also copies B's own teacher arm, as arm NAME: a reference run at another time step (A, e.g. the
teacher at 0.5 fs) scored against B's teacher and arms alike (2026-11-15, the water film on Cu(111): is the 1 fs teacher
itself heating the film?).
"""

from __future__ import annotations

import sys

import numpy as np


def main(argv) -> int:
    keep = None
    if "--keep-teacher" in argv:
        i = argv.index("--keep-teacher")
        keep = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    a, b = np.load(argv[0]), np.load(argv[1])
    suffix, out = argv[2], argv[3]
    for k in ("numbers", "masses", "v_fs", "x_fs"):
        if not np.array_equal(a[k], b[k]):
            raise SystemExit(f"{k} differs between the two runs")
    if not np.allclose(a["cell"], b["cell"]):
        raise SystemExit("cell differs")
    merged = {k: a[k] for k in ("numbers", "cell", "masses", "v_fs", "x_fs")}
    merged["teacher__v"], merged["teacher__x"] = a["teacher__v"], a["teacher__x"]
    for k in b.files:
        if "__" in k and not k.startswith("teacher__"):
            arm, kind = k.split("__")
            if b[k].shape[0] != merged[f"teacher__{kind}"].shape[0]:
                raise SystemExit(f"{k}: {b[k].shape[0]} frames against the teacher's {merged[f'teacher__{kind}'].shape[0]}")
            merged[f"{arm}_{suffix}__{kind}"] = b[k]
    if keep is not None:
        for kind in ("v", "x"):
            if b[f"teacher__{kind}"].shape[0] != merged[f"teacher__{kind}"].shape[0]:
                raise SystemExit(f"B's teacher {kind}: {b[f'teacher__{kind}'].shape[0]} frames against "
                                 f"{merged[f'teacher__{kind}'].shape[0]}")
            merged[f"{keep}__{kind}"] = b[f"teacher__{kind}"]
    np.savez(out, **merged)
    print(f"-> {out}: " + ", ".join(sorted({k.split('__')[0] for k in merged if '__' in k})))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
