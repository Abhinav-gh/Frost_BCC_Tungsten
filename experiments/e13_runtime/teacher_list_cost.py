"""Is the teacher baseline at its strongest? The cost of the Verlet skin in the teacher's edge list (2026-11-16).

Every teacher call in Frost, the baseline's included, is the graphed MACE call on a Verlet list built at r_max + skin
(``frost.runtime.md.build``: skin 1.0 A), rebuilt only when an atom has moved half the skin. MACE evaluates radial
features and tensor products on every edge it is given and zeroes those beyond r_max only through its envelope, so the
skin's pairs cost compute: for fcc Cu with MACE-MP-0 (r_max 6 A) a 7 A list holds 134 neighbours per atom against 78
within 6 A. The spatial region profile found the same on a cut-out (135,022 edges with the skin, 55.6 ms; a tight list on
the same atoms, 82,518 edges, 35.2 ms). If a tight list rebuilt for every call costs less than the skin's edges, the
strongest baseline (trap 22: every speedup is quoted against the strongest stack) rebuilds a tight list each step.

Per system, at its start frame, on one GPU in float32 (cuEquivariance, CUDA graph, edge forces, as ``build`` runs it):
- the graphed teacher call on the skin list (1.0 A, as used) and on a tight list (skin 0), edge counts, the forces'
  agreement;
- the cost of building the tight list (NVIDIA ALCHEMI's GPU cell list, as the runtime builds lists), and of writing it
  into the graph's buffers (``set_edges``) without a recapture;
- so the per-step cost of the tight-list baseline: call + build + write, against the skin baseline's call.

    PYTHONPATH=src python experiments/e13_runtime/teacher_list_cost.py [--runs cu500_T300,silica1152_T300,...]
Output: data/results/e13/teacher_list_cost.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from ase import Atoms

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "results" / "e13"


def cuda_ms(fn, reps: int = 30) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="cu500_T300,li3po4_T300,silica1152_T300,water216off_T300,cuwaterw_T300")
    ap.add_argument("--frame", type=int, default=9000)
    args = ap.parse_args()
    from frost.neighbours import VerletList
    from frost.teacher_fast import GraphedTeacher
    from frost.teachers import load_calc, teacher_of

    dev = "cuda"
    torch.set_default_dtype(torch.float32)
    out = {"gpu": torch.cuda.get_device_name(0), "when": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "rows": []}
    calcs = {}
    for run in args.runs.split(","):
        with h5py.File(ROOT / "data" / "trajectories" / f"{run}.h5", "r") as f:
            name = teacher_of(f.attrs)
            numbers = np.array(f.attrs["numbers"])
            cell = np.array(f.attrs["cell"])
            frame = min(args.frame, f["positions"].shape[0] - 1)
            pos = np.array(f["positions"][frame], dtype=np.float64)
        atoms = Atoms(numbers=numbers, positions=pos, cell=cell, pbc=True)
        if name not in calcs:
            calcs[name] = load_calc(name, dev, "float32", enable_cueq=True)
        calc = calcs[name]
        r_max = float(calc.models[0].r_max)
        x = torch.as_tensor(pos, dtype=torch.float32, device=dev)
        row = {"run": run, "teacher": name, "atoms": len(atoms), "r_max": r_max}
        try:
            lists, forces = {}, {}
            for label, skin in (("skin", 1.0), ("tight", 0.0)):
                vl = VerletList(cell, r_max, skin, dev, wrap_positions=False, builder="alchemi")
                vl.update(x, force=True)
                gt = GraphedTeacher(calc, atoms, dev, edge_forces=True)
                gt.set_edges(vl.i, vl.j, vl.shifts, vl.unit_shifts)
                e, f_ = gt(x)
                forces[label] = f_.detach().double().clone()
                row[f"{label}_edges"] = int(vl.i.numel())
                row[f"{label}_call_ms"] = cuda_ms(lambda: gt(x))
                lists[label] = (vl, gt)
            row["max_dF"] = float((forces["skin"] - forces["tight"]).abs().max())
            vl, gt = lists["tight"]

            def build():
                vl.update(x, force=True)

            def build_and_write():
                vl.update(x, force=True)
                gt.set_edges(vl.i, vl.j, vl.shifts, vl.unit_shifts)
                return gt(x)

            row["tight_build_ms"] = cuda_ms(build)
            row["tight_build_write_call_ms"] = cuda_ms(build_and_write)
            row["captures_after"] = int(gt.captures)
            row["tight_step_over_skin_call"] = row["tight_build_write_call_ms"] / row["skin_call_ms"]
            # the runtime's tight mode (md.build tight_teacher): keep the skin list, filter it to r_max for every call
            from frost.runtime.md import tight_edges

            vs, _ = lists["skin"]
            gf = GraphedTeacher(calc, atoms, dev, edge_forces=True)
            gf.set_edges(*tight_edges(vs, x, r_max ** 2))
            f_filt = gf(x)[1].detach().double().clone()
            row["filtered_max_dF"] = float((f_filt - forces["skin"]).abs().max())

            def filtered_step():
                vs.update(x)  # no rebuild at a fixed frame: the common call
                gf.set_edges(*tight_edges(vs, x, r_max ** 2))
                return gf(x)

            row["filter_write_ms"] = cuda_ms(lambda: gf.set_edges(*tight_edges(vs, x, r_max ** 2)))
            row["filtered_step_ms"] = cuda_ms(filtered_step)
            row["filtered_step_over_skin_call"] = row["filtered_step_ms"] / row["skin_call_ms"]
            row["filtered_captures"] = int(gf.captures)
            del lists, gf
        except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
            if "out of memory" not in str(err).lower():
                raise
            row["oom"] = True
        out["rows"].append(row)
        print(row, flush=True)
        torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "teacher_list_cost.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(path)
    print("->", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
