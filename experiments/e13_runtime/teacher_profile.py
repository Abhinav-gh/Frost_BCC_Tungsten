"""Where the teacher's time goes, and what public switches buy (MD e2e profile, the teacher side).

The teacher (MACE-MP-0 medium, cuEquivariance with conv fusion, captured in a CUDA graph) costs
42-48 us per atom-step on an A6000 at 500 and 13,500 atoms alike, so it is compute-bound, and at
k = 24 it is half of Frost's step on Cu. This times the teacher force call per run in each variant:
- ``base``: as every arm runs it now (``GraphedTeacher`` on ``mace_mp(enable_cueq=True)``);
- ``compile``: the same with mace's own ``compile_mode="default"`` constructor argument;
- ``tf32``: the documented ``torch.backends.cuda.matmul.allow_tf32`` flag (changes the numerics);
- ``compile+tf32``;
- ``edge``: forces from mace's public ``compute_edge_forces`` output summed onto atoms with ``index_add_``
  (``GraphedTeacher(edge_forces=True)``), which skips the sort-based backward of the edge gather.
Each variant's forces are compared with ``base`` on the same frame (max and RMS difference), so a
variant that changes the physics is visible next to its time. ``--kernels`` adds a torch.profiler
table of the base call's top kernels (uncaptured, so each kernel is visible).

Output: data/results/e13/teacher_profile_<tag>.json

Usage (guarded launcher; GPU)
  scripts/safe_run.sh teacher_profile .venvs/frost-mace/bin/python experiments/e13_runtime/teacher_profile.py \\
      --runs cu500_T300,li3po4_T300 --kernels --tag v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TRAJ = ROOT / "data" / "trajectories"
RES = ROOT / "data" / "results" / "e13"


def frame(run: str, index: int):
    from frost.teacher import frame_atoms

    with h5py.File(TRAJ / f"{run}.h5", "r") as fh:
        numbers = np.asarray(fh.attrs["numbers"], dtype=np.int64)
        cell = np.asarray(fh.attrs["cell"], dtype=np.float64)
        positions = np.asarray(fh["positions"][index], dtype=np.float64)
    return frame_atoms(numbers, positions, cell), cell


class ModelHolder:
    """What ``GraphedTeacher`` reads from a calculator (``models[0]``), holding a model built by mace's own converter."""

    def __init__(self, model):
        self.models = [model]


def build(atoms, cell, compile_mode: str | None, edge_forces: bool = False, kernels: str = "cueq"):
    from mace.calculators import mace_mp

    from frost.neighbours import VerletList
    from frost.teacher_fast import GraphedTeacher

    calc = mace_mp(model="medium", device="cuda", default_dtype="float32", enable_cueq=kernels == "cueq",
                   compile_mode=compile_mode)
    if kernels == "oeq":  # OpenEquivariance through mace's public converter (mace.cli.convert_e3nn_oeq)
        from mace.cli.convert_e3nn_oeq import run as to_oeq

        calc = ModelHolder(to_oeq(calc.models[0], device="cuda"))
    teacher = GraphedTeacher(calc, atoms, "cuda", edge_forces=edge_forces)
    x = torch.as_tensor(atoms.get_positions(), dtype=torch.float64, device="cuda")
    lst = VerletList(cell, teacher.r_max, 1.0, "cuda", wrap_positions=False, builder="alchemi")
    lst.update(x, force=True)
    teacher.set_edges(lst.i, lst.j, lst.shifts, lst.unit_shifts)
    return teacher, x, lst


def time_ms(fn, repeats: int = 50, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(repeats):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b))
    return float(np.median(out))


def kernel_table(teacher, x, rows: int = 25) -> list[dict]:
    """The uncaptured force call's top kernels by GPU time."""
    from torch.profiler import ProfilerActivity, profile

    data = dict(teacher.data)
    data["positions"] = x.to(teacher.dtype).clone()
    for _ in range(3):
        teacher.model(data, compute_force=True, training=False)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            teacher.model(data, compute_force=True, training=False)
        torch.cuda.synchronize()
    events = [e for e in prof.key_averages() if e.device_time_total > 0]
    total = sum(e.self_device_time_total for e in events)
    events.sort(key=lambda e: -e.self_device_time_total)
    out = []
    for e in events[:rows]:
        out.append({"name": e.key[:90], "calls": e.count // 5, "ms_per_call": e.self_device_time_total / 5e3,
                    "share": e.self_device_time_total / total})
    out.append({"name": "TOTAL", "ms_per_call": total / 5e3, "kernels_per_call": sum(e.count for e in events) // 5})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="cu500_T300,li3po4_T300")
    ap.add_argument("--variants", default="base,edge,edge+tf32")
    ap.add_argument("--index", type=int, default=9000)
    ap.add_argument("--kernels", action="store_true")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    out = {}
    for run in args.runs.split(","):
        atoms, cell = frame(run, args.index)
        n = len(atoms)
        rows, f_base = {}, None
        for variant in args.variants.split(","):
            torch.backends.cuda.matmul.allow_tf32 = "tf32" in variant
            torch.backends.cudnn.allow_tf32 = "tf32" in variant
            try:
                kern = "oeq" if "oeq" in variant else ("e3nn" if "e3nn" in variant else "cueq")
                teacher, x, lst = build(atoms, cell, "default" if "compile" in variant else None, "edge" in variant,
                                        kern)
                ms = time_ms(lambda: teacher(x))
                f = teacher(x)[1].detach().double().clone()
            except Exception as exc:  # a variant the stack cannot run is a result, not a crash
                rows[variant] = {"error": repr(exc)[:300]}
                print(f"{run} {variant:13s} failed: {exc!r}"[:300], flush=True)
                continue
            if f_base is None:
                f_base = f
            d = (f - f_base).norm(dim=1)
            rows[variant] = {"ms": ms, "us_per_atom": 1e3 * ms / n, "edges": int(lst.i.numel()),
                             "max_dF_meV_A": 1e3 * float(d.max()), "rms_dF_meV_A": 1e3 * float(d.pow(2).mean().sqrt())}
            print(f"{run} {variant:13s} {ms:7.3f} ms  {1e3 * ms / n:6.2f} us/atom  edges {lst.i.numel()}  "
                  f"dF max {rows[variant]['max_dF_meV_A']:.3f} rms {rows[variant]['rms_dF_meV_A']:.3f} meV/A", flush=True)
            if args.kernels and variant == "base":
                rows["kernels"] = kernel_table(teacher, x)
                for r in rows["kernels"]:
                    print(f"   {r.get('share', 1.0):6.1%} {r['ms_per_call']:7.3f} ms  x{r.get('calls', '')}  {r['name']}",
                          flush=True)
            del teacher
            torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        out[run] = {"n_atoms": n, "variants": rows}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"teacher_profile_{args.tag}.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
