"""Can the long-range part of a MACELES teacher be held? (further_expansions_beyond.md 6.8: the third level.)

``frost.les_split.HeldLongRange`` holds what the Ewald sum produces, the potential phi = dE_lr/dq and the direct force
g = -dE_lr/dx at fixed charges, and recomputes the charge path through the network on every call. This measures, for the
released MACELES-OFF small on cuEquivariance kernels, float32:

1. correctness: the composed exact mode against MACELES's own forces, and the held mode at its refresh point;
2. the hold error on water (648 atoms, frames 1 fs apart from the MACE-OFF23 trajectory): refresh at t0, then the held
   force at t0 + d against the exact force, d = 1-32 fs, RMS per species in the certificate's units, meV/(A amu), for real
   masses and for hydrogen mass repartitioning by 3 amu (the water runs' protocol, where epsilon is 6.31); beside it,
   holding the whole long-range force (the short-range network plus F_lr(t0)) and the exact force's own change;
3. the cost: MACELES's eager call, the held call eager and inside a CUDA graph (``frost.teacher_fast.GraphedTeacher``),
   a refresh (charges by a no-grad forward plus the Ewald sum on leaves) and ``exact_and_refresh``, on 648, 5,184 and
   17,496 atoms (the water box repeated).

    PYTHONPATH=src:.tools/les-overlay python experiments/e22_longrange/les_hold.py [--skip-timing]
Output: data/results/e22/les_hold.json
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
from mace import data as mdata
from mace.tools import torch_geometric, utils

from frost.les_split import HeldLongRange, fresh_maceles

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "third_party" / "les_fit_models" / "MACELES-OFF_small_converted.model"
TRAJ = ROOT / "data" / "trajectories" / "water216off_T300.h5"
OUT = ROOT / "data" / "results" / "e22"


def batch_for(atoms: Atoms, model, dev: str) -> dict:
    z = utils.AtomicNumberTable([int(v) for v in model.atomic_numbers])
    cfg = mdata.config_from_atoms(atoms)
    ad = mdata.AtomicData.from_config(cfg, z_table=z, cutoff=float(model.r_max), heads=["Default"])
    d = torch_geometric.Batch.from_data_list([ad]).to(dev).to_dict()
    for k, v in list(d.items()):
        if torch.is_tensor(v) and torch.is_floating_point(v):
            d[k] = v.float()
    return d


def cuda_ms(fn, reps: int = 10) -> float:
    for _ in range(3):
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
    ap.add_argument("--skip-timing", action="store_true")
    ap.add_argument("--starts", type=int, default=20)
    args = ap.parse_args()
    dev = "cuda"
    torch.set_default_dtype(torch.float32)
    model = fresh_maceles(MODEL, dev, "cueq")
    held = HeldLongRange(model)
    short = HeldLongRange(model)
    short.mode = "short"
    with h5py.File(TRAJ, "r") as f:
        cell = np.array(f.attrs["cell"])
        numbers = np.array(f.attrs["numbers"])
        frames = f["positions"]
        n_frames = frames.shape[0]
        starts = np.linspace(1000, n_frames - 40, args.starts).astype(int)
        lags = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
        need = sorted({int(s + d) for s in starts for d in (0, *lags)})
        xs = {i: np.array(frames[i], dtype=np.float64) for i in need}
    res = {"model": str(MODEL.relative_to(ROOT)), "gpu": torch.cuda.get_device_name(0), "dtype": "float32",
           "when": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}

    def atoms_at(i):
        return Atoms(numbers=numbers, positions=xs[i], cell=cell, pbc=True)

    # 1. correctness
    d0 = batch_for(atoms_at(int(starts[0])), model, dev)
    f_ref = model(dict(d0), compute_force=True)["forces"].detach()
    held.mode = "exact"
    f_exact = held(dict(d0))["forces"].detach()
    held.refresh(dict(d0))
    held.mode = "held"
    f_held0 = held(dict(d0))["forces"].detach()
    both = held.exact_and_refresh(dict(d0))["forces"].detach()
    scale = float(f_ref.abs().max())
    res["check"] = {"max_force": scale, "exact_vs_maceles": float((f_exact - f_ref).abs().max()),
                    "held_at_refresh_vs_maceles": float((f_held0 - f_ref).abs().max()),
                    "exact_and_refresh_vs_maceles": float((both - f_ref).abs().max())}
    print("check:", res["check"], flush=True)

    # 2. the hold error over d fs, per species, acceleration units
    masses = {"real": np.where(numbers == 1, 1.008, 15.999), "hmr3": np.where(numbers == 1, 4.008, 12.999)}
    species = {"H": numbers == 1, "O": numbers == 8}
    acc: dict[str, dict] = {}
    for s in starts:
        s = int(s)
        d_s = batch_for(atoms_at(s), model, dev)
        held.refresh(dict(d_s))
        f0 = model(dict(d_s), compute_force=True)["forces"].detach()
        f_lr0 = f0 - short(dict(d_s))["forces"].detach()
        for lag in lags:
            d_t = batch_for(atoms_at(s + lag), model, dev)
            f_t = model(dict(d_t), compute_force=True)["forces"].detach()
            held.mode = "held"
            e_held = (held(dict(d_t))["forces"].detach() - f_t).double().cpu().numpy()
            e_whole = (short(dict(d_t))["forces"].detach() + f_lr0 - f_t).double().cpu().numpy()
            e_force = (f_t - f0).double().cpu().numpy()
            for mk, m in masses.items():
                for name, err in (("held", e_held), ("whole", e_whole), ("force_change", e_force)):
                    a = 1e3 * err / m[:, None]  # meV / (A amu)
                    for sp, mask in species.items():
                        acc.setdefault(f"{mk}:{name}:{sp}", {}).setdefault(lag, []).append(
                            float(np.sqrt((a[mask] ** 2).mean())))
                    acc.setdefault(f"{mk}:{name}:all", {}).setdefault(lag, []).append(float(np.sqrt((a ** 2).mean())))
    res["hold"] = {k: {str(lag): float(np.mean(v)) for lag, v in sorted(rows.items())} for k, rows in acc.items()}
    for key in ("hmr3:held:all", "hmr3:whole:all", "hmr3:force_change:all", "hmr3:held:H", "hmr3:held:O"):
        print(key, " ".join(f"{lag}fs {val:.3f}" for lag, val in res["hold"][key].items()), flush=True)

    # 3. the cost
    if not args.skip_timing:
        from frost.neighbours import edges_for
        from frost.teacher_fast import GraphedTeacher

        class Calc:  # what GraphedTeacher reads: models[0]
            models = [held]

        rows = []
        base = atoms_at(int(starts[0]))
        for rep in (1, 2, 3):
            atoms = base.repeat((rep, rep, rep))
            label = f"water x{rep}^3 ({len(atoms):,} atoms)"
            d = batch_for(atoms, model, dev)
            row = {"system": label, "atoms": len(atoms)}
            try:
                row["maceles_eager_ms"] = cuda_ms(lambda: model(dict(d), compute_force=True)["forces"])
                held.mode = "exact"
                row["composed_exact_eager_ms"] = cuda_ms(lambda: held(dict(d))["forces"])
                held.refresh(dict(d))
                held.mode = "held"
                row["held_eager_ms"] = cuda_ms(lambda: held(dict(d))["forces"])
                row["refresh_ms"] = cuda_ms(lambda: held.refresh(dict(d)))
                row["exact_and_refresh_ms"] = cuda_ms(lambda: held.exact_and_refresh(dict(d))["forces"])
                gt = GraphedTeacher(Calc(), atoms, device=dev)
                x = torch.as_tensor(atoms.get_positions(), dtype=torch.float32, device=dev)
                xo, i, j, sh, us = edges_for(atoms.get_positions(), np.array(atoms.cell), float(model.r_max),
                                             device=dev, dtype=torch.float32, wrap_positions=False)
                held.mode = "held"
                gt.set_edges(i, j, sh, us)
                e_g, f_g = gt(x)
                f_cmp = held(dict(d))["forces"].detach()
                row["graphed_vs_eager_max"] = float((f_g - f_cmp).abs().max())
                row["held_graphed_ms"] = cuda_ms(lambda: gt(x))
                row["refresh_plus_graphed_ms"] = cuda_ms(lambda: (held.refresh(gt.data), gt(x)))
                del gt
            except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
                if "out of memory" not in str(err).lower():
                    raise
                row["oom"] = True
            rows.append(row)
            print(row, flush=True)
            torch.cuda.empty_cache()
        res["timing"] = rows
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "les_hold.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(res, indent=1))
    tmp.replace(path)
    print("->", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
