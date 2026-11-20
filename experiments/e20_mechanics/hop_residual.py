"""Does the student underestimate a screw core's hop barrier? (2026-11-18, the dislocation over-glide)

Both multirate arms move the cores about twice as far as the teacher (summed net glide 299 / 292 A against 140 at 1500 K,
x1.4 at 1000 K), and the region clause never bound (the runs were identical with it), so the per-interval force change at
the core is not what differs. If the student's energy is too low where the core sits between Peierls valleys, the inner
steps cross on the student's lower barrier, and a residual held from the valley cannot follow a hop that lasts about one
interval. This measures it on the teacher's own frames: cores tracked (screw_cores), hop frames found (a core moving at
least ``--hop-a`` within two frames), and on each hop's window (frames t - w .. t + w) and on as many quiet windows the
teacher and the student give energies; the statistic is the change of E_teacher - E_student from the window's first
frame to its middle, hops against quiet. A lower student barrier shows as a positive hop-minus-quiet difference.

    python experiments/e20_mechanics/hop_residual.py data/results/e13/frames_<run>_<tag>.npz [--arm teacher]
Output: data/results/e20/hop_residual_<stem>.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "results" / "e20"


def _cores_module():
    spec = importlib.util.spec_from_file_location("screw_cores", Path(__file__).with_name("screw_cores.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def core_paths(x: np.ndarray, cell: np.ndarray):
    """Per frame, the + and - core x positions (unwrapped), from the Burgers circuits."""
    sc = _cores_module()
    p0, col_of, cols, tris, centroids, b = sc.setup(cell, x.shape[1])
    lx = cell[0, 0]
    paths = {+1: [], -1: []}
    for f in range(len(x)):
        circ = sc.circulation(x[f], p0, col_of, cols, tris, cell, b)
        for sign in (+1, -1):
            hit = np.flatnonzero(sign * circ > 0.5 * b)
            if len(hit):
                cx = centroids[hit, 0]
                cx = cx[0] + np.mean(cx - cx[0] - lx * np.round((cx - cx[0]) / lx))
            else:
                cx = paths[sign][-1] if paths[sign] else np.nan
            if paths[sign]:
                prev = paths[sign][-1]
                cx = prev + (cx - prev) - lx * np.round((cx - prev) / lx)
            paths[sign].append(cx)
    return {s: np.array(v) for s, v in paths.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames")
    ap.add_argument("--arm", default="teacher")
    ap.add_argument("--student", default="", help="student checkpoint (default: the run's, from its md_ result)")
    ap.add_argument("--hop-a", type=float, default=2.0, help="A: a hop moves a core this far within two frames")
    ap.add_argument("--window", type=int, default=3, help="frames on each side of a hop")
    ap.add_argument("--max-events", type=int, default=40)
    args = ap.parse_args()
    from frost.neighbours import VerletList, tight_edges
    from frost.student import GraphedStudent
    from frost.students import load_student
    from frost.teacher import frame_atoms
    from frost.teacher_fast import GraphedTeacher
    from frost.teachers import load_calc, teacher_of

    path = Path(args.frames)
    d = np.load(path)
    stem = path.stem.removeprefix("frames_")
    run, tag = stem.split("_dfx")[0], "dfx" + stem.split("_dfx")[1] if "_dfx" in stem else stem
    cfg = json.loads((ROOT / "data" / "results" / "e13" / f"md_{stem}.json").read_text())["config"]
    student_path = args.student or cfg["student"]
    import h5py

    with h5py.File(ROOT / "data" / "trajectories" / f"{cfg['run']}.h5", "r") as fh:
        teacher_name = teacher_of(fh.attrs)
    cell = np.asarray(d["cell"], dtype=np.float64)
    numbers = d["numbers"]
    x = np.asarray(d[f"{args.arm}__x"], dtype=np.float64)
    paths = core_paths(x, cell)
    w = args.window
    hops = []
    for sign, p in paths.items():
        jump = np.abs(p[2:] - p[:-2])
        f = 1
        while f < len(jump) - 1:
            if jump[f] >= args.hop_a and w <= f + 1 < len(x) - w:
                hops.append((sign, f + 1))
                f += 2 * w
            f += 1
    rng = np.random.default_rng(0)
    hop_frames = {h[1] for h in hops}
    quiet = []
    while len(quiet) < min(len(hops), args.max_events) and len(quiet) < 400:
        f = int(rng.integers(w, len(x) - w))
        if all(abs(f - h) > 4 * w for h in hop_frames):
            quiet.append(f)
    hops = hops[: args.max_events]
    dev = "cuda"
    calc = load_calc(teacher_name, dev, "float32", enable_cueq=True)
    atoms = frame_atoms(numbers, x[0], cell)
    teacher = GraphedTeacher(calc, atoms, dev, edge_forces=True)
    st = load_student(student_path, dev)
    types = st.types(torch.as_tensor(numbers, device=dev))
    t_list = VerletList(cell, teacher.r_max, 1.0, dev, wrap_positions=False, builder="alchemi")
    s_cut = float(st.config["cutoff"])
    s_list = VerletList(cell, s_cut, 0.5, dev, wrap_positions=False, builder="alchemi")
    x0 = torch.as_tensor(x[0], dtype=torch.float32, device=dev)
    t_list.update(x0, force=True)
    s_list.update(x0, force=True)
    teacher.set_edges(*tight_edges(t_list, x0, teacher.r_max ** 2))
    student = GraphedStudent(st, types, len(numbers), capacity=int(1.4 * s_list.i.numel()) + 64)
    student.set_edges(s_list.i, s_list.j, s_list.shifts)

    def delta_e(f: int) -> float:
        xf = torch.as_tensor(x[f], dtype=torch.float32, device=dev)
        t_list.update(xf)
        teacher.set_edges(*tight_edges(t_list, xf, teacher.r_max ** 2))
        e_t, _ = teacher(xf)
        s_list.update(xf, force=True)
        student.set_edges(s_list.i, s_list.j, s_list.shifts)
        e_s, _ = student(xf)
        return float(e_t) - float(e_s.detach().sum())

    def window_change(c: int) -> list[float]:
        vals = [delta_e(f) for f in range(c - w, c + w + 1)]
        return [v - vals[0] for v in vals]

    hop_curves = np.array([window_change(c) for _, c in hops]) if hops else np.zeros((0, 2 * w + 1))
    quiet_curves = np.array([window_change(c) for c in quiet]) if quiet else np.zeros((0, 2 * w + 1))
    mid = w
    h_mid = hop_curves[:, mid] if len(hop_curves) else np.array([])
    q_mid = quiet_curves[:, mid] if len(quiet_curves) else np.array([])
    report = {"frames": str(path), "arm": args.arm, "student": student_path, "hops": len(hops), "quiet": len(quiet),
              "x_fs": float(d["x_fs"]), "window_frames": w,
              "hop_mean_curve_eV": hop_curves.mean(0).tolist() if len(hop_curves) else None,
              "quiet_mean_curve_eV": quiet_curves.mean(0).tolist() if len(quiet_curves) else None,
              "hop_mid_mean_eV": float(h_mid.mean()) if len(h_mid) else None,
              "hop_mid_se_eV": float(h_mid.std(ddof=1) / np.sqrt(len(h_mid))) if len(h_mid) > 1 else None,
              "quiet_mid_mean_eV": float(q_mid.mean()) if len(q_mid) else None,
              "quiet_mid_se_eV": float(q_mid.std(ddof=1) / np.sqrt(len(q_mid))) if len(q_mid) > 1 else None}
    print(json.dumps({k: v for k, v in report.items() if "curve" not in k}, indent=1), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"hop_residual_{stem}.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=1))
    tmp.replace(out)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
