"""Where a force call's GPU time goes, by kind of work, for MACE, NequIP, Allegro and the student.

The research-direction review (docs/design/research_direction.md) has to decide two
things from measurement rather than description:

- whether aggregation (the scatter-sum of edge messages onto atoms) costs enough
  that caching it could matter at all, on any of the three architectures;
- which kind of work dominates each architecture, so that a caching unit is chosen
  where the time actually is.

v1's E2 answered a narrower question for MACE by swapping modules for stand-ins.
That swap is a runtime patch, which this project no longer allows, and it cannot
separate aggregation from the tensor product anyway. This script changes nothing
in any model: it runs the unmodified force call (forward and backward) under
torch.profiler and sums each operator's *self* device time into five buckets.

- scatter: index_add, scatter_add, scatter_reduce, segment_reduce, index_put
  (the forward of sum aggregation, and the backward of a gather);
- gather: index_select, index, gather, embedding (the forward of edge <- node,
  and the backward of a scatter);
- tensor product: operators from cuEquivariance, OpenEquivariance or other
  dedicated equivariant kernels, by name;
- dense matmul: mm, addmm, bmm, baddbmm (radial MLPs, linears, and on the e3nn
  and Allegro paths the tensor product itself, which lowers to these);
- other: elementwise work, reductions and copies. On the e3nn and Allegro paths
  this bucket also holds most of the tensor product, which lowers to mul, sum
  and add rather than to a named kernel; the top operators are recorded so the
  reader can see what is in it.

The forward scatter and its backward gather are one aggregation. Both buckets
count towards it: removing an aggregation removes both.

Device time is summed over kernels, so it excludes launch gaps. The eager wall
time of the same call is reported alongside; their difference is launch and host
overhead, which a CUDA graph removes and a cache cannot.

Output: data/results/e9/op_profile_<arch>_si<N>[_<tag>].json

Usage (guarded launcher, GPU, float32)
  scripts/safe_run.sh e9_prof_mace .venvs/frost-mace/bin/python \
      experiments/e9_direction/op_profile.py --arch mace --cells 4
  scripts/safe_run.sh e9_prof_allegro .venvs/frost-nequip/bin/python \
      experiments/e9_direction/op_profile.py --arch allegro --cells 3
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "results" / "e9"
NEQUIP_PKG = {
    "nequip": ROOT / "data" / "models" / "nequip" / "NequIP-MP-L-0.1.nequip.zip",
    "allegro": ROOT / "data" / "models" / "nequip" / "Allegro-MP-L-0.1.nequip.zip",
}
STUDENT = ROOT / "data" / "models" / "students" / "si512_T300_s35c16e1.pt"

SCATTER = ("index_add", "scatter_add", "scatter_reduce", "segment_reduce", "index_put",
           "scatter")
GATHER = ("index_select", "gather", "embedding", "take")
TP_NAMES = ("cuequivariance", "segmented", "tensor_product", "openequivariance", "oeq")
MATMUL = ("aten::mm", "aten::addmm", "aten::bmm", "aten::baddbmm", "aten::_addmm_activation")


def bucket(key: str) -> str:
    k = key.lower()
    if any(s in k for s in TP_NAMES):
        return "tensor_product"
    if any(s in k for s in SCATTER):
        return "scatter"
    if any(s in k for s in GATHER) or k == "aten::index":
        return "gather"
    if key in MATMUL:
        return "dense_matmul"
    return "other"


def silicon(cells: int, rattle: float, seed: int):
    from frost.systems import si_scaling

    atoms = si_scaling(cells)
    rng = np.random.default_rng(seed)
    atoms.positions += rng.normal(0.0, rattle, atoms.positions.shape)
    return atoms


def build_mace(atoms, device: str, cueq: bool):
    from mace.calculators import mace_mp

    from frost.calibrate import _batch_for

    calc = mace_mp(model="medium", device=device, default_dtype="float32", enable_cueq=cueq)
    model = calc.models[0]
    data = _batch_for(calc, atoms, float(model.r_max), device)[0].to_dict()
    n_edges = int(data["edge_index"].shape[1])

    def call():
        return model(data, compute_force=True, training=False)["forces"]

    return call, n_edges


def build_student(atoms, device: str, backend: str):
    from frost.neighbours import VerletList
    from frost.student import energy_and_forces
    from frost.students import load_student

    student = load_student(STUDENT, device)
    if student.config.get("kind") == "mace" and backend != student.config.get("backend", "e3nn"):
        student = student.converted(backend)
    types = student.types(torch.as_tensor(atoms.numbers, device=device))
    cell = np.array(atoms.cell, dtype=np.float64)
    x = torch.as_tensor(atoms.get_positions(), dtype=torch.float64, device=device)
    lst = VerletList(cell, float(student.config["cutoff"]), 0.0, device, wrap_positions=False)
    lst.update(x, force=True)
    xs = x.to(torch.float32)

    def call():
        return energy_and_forces(student, xs, types, lst.i, lst.j, lst.shifts.to(xs.dtype))[1]

    return call, int(lst.i.numel()), student.config


def build_nequip(arch: str, atoms, device: str):
    from nequip.data import AtomicDataDict, from_ase
    from nequip.data import transforms as T  # noqa: N812
    from nequip.model import ModelFromPackage, ModelTypeNamesFromPackage

    pkg = str(NEQUIP_PKG[arch])
    names = ModelTypeNamesFromPackage(pkg)
    model = ModelFromPackage(package_path=pkg)["sole_model"].eval().to(device)
    mp = inspect.signature(T.ChemicalSpeciesToAtomTypeMapper).parameters
    kw = {}
    if "model_type_names" in mp:
        kw["model_type_names"] = names
    if "chemical_species_to_atom_type_map" in mp:
        kw["chemical_species_to_atom_type_map"] = {n: n for n in names}
    r_max = 6.0
    nl_params = inspect.signature(T.NeighborListTransform).parameters
    nl_kw = {"r_max": r_max} if "r_max" in nl_params else {next(iter(nl_params)): r_max}
    data = from_ase(atoms)
    for t in (T.ChemicalSpeciesToAtomTypeMapper(**kw), T.NeighborListTransform(**nl_kw)):
        data = t(data)
    data = AtomicDataDict.to_(data, torch.device(device))
    n_edges = int(data[AtomicDataDict.EDGE_INDEX_KEY].shape[1])

    def call():
        return model(dict(data))[AtomicDataDict.FORCE_KEY]

    return call, n_edges


def wall_ms(call, repeats: int, warmup: int) -> float:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        call()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(times))


def profile(call, repeats: int) -> tuple[dict, list]:
    from torch.profiler import ProfilerActivity
    from torch.profiler import profile as tprofile

    call()
    torch.cuda.synchronize()
    with tprofile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(repeats):
            call()
        torch.cuda.synchronize()
    from torch.autograd import DeviceType

    per_op: dict[str, float] = {}
    for ev in prof.key_averages():
        # kernels appear twice: as their own events and inside the operator that
        # launched them. Count operators only, so each kernel is summed once.
        if ev.device_type != DeviceType.CPU:
            continue
        t = getattr(ev, "self_device_time_total", None)
        if t is None:
            t = getattr(ev, "self_cuda_time_total", 0.0)
        if t and t > 0:
            per_op[ev.key] = per_op.get(ev.key, 0.0) + float(t) / 1e3 / repeats  # ms per call
    buckets: dict[str, float] = {}
    for key, ms in per_op.items():
        b = bucket(key)
        buckets[b] = buckets.get(b, 0.0) + ms
    top = sorted(per_op.items(), key=lambda kv: -kv[1])[:20]
    return buckets, [{"op": k, "bucket": bucket(k), "ms": round(v, 4)} for k, v in top]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", required=True, choices=["mace", "mace-e3nn", "nequip", "allegro",
                                                      "student"])
    ap.add_argument("--cells", type=int, default=4, help="diamond Si cells per side (8 atoms each)")
    ap.add_argument("--rattle", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--student-backend", default="cueq")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if not args.device.startswith("cuda"):
        raise SystemExit("GPU only: these models are not run on CPU on this server")

    atoms = silicon(args.cells, args.rattle, args.seed)
    extra: dict = {}
    if args.arch in ("mace", "mace-e3nn"):
        call, n_edges = build_mace(atoms, args.device, cueq=args.arch == "mace")
    elif args.arch == "student":
        call, n_edges, cfg = build_student(atoms, args.device, args.student_backend)
        extra["student_config"] = {k: v for k, v in cfg.items() if not isinstance(v, list)}
    else:
        call, n_edges = build_nequip(args.arch, atoms, args.device)

    forces = call().detach()
    wall = wall_ms(call, args.repeats, warmup=5)
    buckets, top = profile(call, args.repeats)
    device_ms = sum(buckets.values())
    shares = {k: round(v / device_ms, 4) for k, v in sorted(buckets.items())}
    aggregation = shares.get("scatter", 0.0) + shares.get("gather", 0.0)

    report = {
        "arch": args.arch,
        "n_atoms": len(atoms),
        "n_edges": n_edges,
        "force_rms_eV_per_A": float(forces.norm(dim=1).pow(2).mean().sqrt()),
        "wall_ms_eager": round(wall, 3),
        "device_ms": round(device_ms, 3),
        "launch_and_host_share": round(max(0.0, 1.0 - device_ms / wall), 4),
        "bucket_ms": {k: round(v, 4) for k, v in sorted(buckets.items())},
        "bucket_share_of_device": shares,
        "aggregation_share_of_device": round(aggregation, 4),
        "top_ops": top,
        **extra,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    path = OUT / f"op_profile_{args.arch}_si{len(atoms)}{tag}.json"
    path.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "top_ops"}, indent=1))
    for row in top[:12]:
        print(f"  {row['ms']:9.4f} ms  {row['bucket']:14s} {row['op']}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
