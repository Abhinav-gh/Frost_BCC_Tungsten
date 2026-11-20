"""Kanda et al., multiscale multi-timestep switching (Adv. Intell. Discov. 2026, e70152), on their own stack: DeePMD-kit.

Their method: two DeePMD models of the same system, a cheap short-range one (4 A) and an accurate long-range one (6 A);
the dynamics use the cheap model's force on most steps and the accurate model's force every ``--every`` th step
(1:3 in their paper). Their code is not public; their stack is (DeePMD-kit, LAMMPS). Here the models are trained with
DeePMD-kit's own trainer on our teacher's labels and evaluated through DeePMD-kit's public inference API
(``deepmd.infer.DeepPot``); the switching schedule is the only code of ours, a velocity Verlet loop that picks the model
per step. The accurate model run every step, through the same loop, is the speed and fidelity reference.

Stages (each in its environment):
- ``label`` (frost-mace): our MACE-MP-0 medium labels on frames of one of our trajectories, in DeePMD's npy format;
- ``config``: DeePMD-kit training inputs (se_e2_a, 4 A and 6 A), from their water example's settings;
- ``md`` (frost-deepmd): the three arms (accurate every step, switching, cheap alone), NVE from our start frame,
  frames saved as ``flashmd_run.py`` saves them, for ``flashmd_score.py``-style scoring against the accurate arm.

Training itself is ``dp --pt train`` / ``dp --pt freeze`` (DeePMD-kit's CLI), run by the chain script.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRAJ = ROOT / "data" / "trajectories"
BASE = ROOT / "data" / "baselines" / "kanda"
RES = ROOT / "data" / "results" / "e14"


def stage_label(args) -> int:
    import torch

    from frost.teacher import frame_atoms, mace_forces
    from frost.teachers import load_calc

    out = BASE / args.run
    with h5py.File(TRAJ / f"{args.run}.h5", "r") as fh:
        numbers = np.asarray(fh.attrs["numbers"])
        cell = np.asarray(fh.attrs["cell"], dtype=np.float64)
        frames = list(range(0, fh["positions"].shape[0], args.stride))
        pos = [np.asarray(fh["positions"][i], dtype=np.float64) for i in frames]
    calc = load_calc("mp-medium", "cuda", "float32", enable_cueq=True)
    species = sorted(set(int(z) for z in numbers))
    from ase.data import chemical_symbols

    type_map = [chemical_symbols[z] for z in species]
    types = np.array([species.index(int(z)) for z in numbers])
    coords, forces, energies = [], [], []
    for x in pos:
        f, e, _ = mace_forces(calc, frame_atoms(numbers, x, cell), "cuda")
        coords.append(x.reshape(-1))
        forces.append(np.asarray(f, dtype=np.float64).reshape(-1))
        energies.append(float(e))
    torch.cuda.empty_cache()
    rng = np.random.default_rng(0)
    order = rng.permutation(len(frames))
    n_val = max(20, len(frames) // 10)
    for split, idx in (("valid", order[:n_val]), ("train", order[n_val:])):
        d = out / split
        (d / "set.000").mkdir(parents=True, exist_ok=True)
        (d / "type.raw").write_text("\n".join(str(t) for t in types) + "\n")
        (d / "type_map.raw").write_text("\n".join(type_map) + "\n")
        np.save(d / "set.000" / "coord.npy", np.asarray(coords)[idx])
        np.save(d / "set.000" / "force.npy", np.asarray(forces)[idx])
        np.save(d / "set.000" / "energy.npy", np.asarray(energies)[idx])
        np.save(d / "set.000" / "box.npy", np.tile(cell.reshape(1, 9), (len(idx), 1)))
    (out / "labels.json").write_text(json.dumps({"run": args.run, "frames": len(frames), "stride": args.stride,
                                                 "type_map": type_map, "teacher": "mace-mp-0 medium"}, indent=1))
    print(f"{len(frames)} frames labelled -> {out} (type map {type_map})", flush=True)
    return 0


def stage_config(args) -> int:
    out = BASE / args.run
    meta = json.loads((out / "labels.json").read_text())
    for rcut, sel in ((4.0, args.sel4), (6.0, args.sel6)):
        cfg = {
            "model": {"type_map": meta["type_map"],
                      "descriptor": {"type": "se_e2_a", "sel": [sel], "rcut_smth": 0.5, "rcut": rcut,
                                     "neuron": [25, 50, 100], "resnet_dt": False, "axis_neuron": 16,
                                     "type_one_side": True, "seed": 1},
                      "fitting_net": {"neuron": [240, 240, 240], "resnet_dt": True, "seed": 1}},
            "learning_rate": {"type": "exp", "decay_steps": 5000, "start_lr": 0.001, "stop_lr": 3.51e-8},
            "loss": {"type": "ener", "start_pref_e": 0.02, "limit_pref_e": 1, "start_pref_f": 1000, "limit_pref_f": 1},
            "training": {"training_data": {"systems": [str(out / "train")], "batch_size": 1},
                         "validation_data": {"systems": [str(out / "valid")], "batch_size": 1, "numb_btch": 3},
                         "numb_steps": args.steps, "seed": 10, "disp_file": "lcurve.out", "disp_freq": 1000,
                         "save_freq": 10000, "stat_file": str(out / f"stat_r{int(rcut)}.hdf5")}}
        d = out / f"r{int(rcut)}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "input.json").write_text(json.dumps(cfg, indent=1))
        print(f"wrote {d / 'input.json'} (rcut {rcut}, sel {sel}, {args.steps} steps)", flush=True)
    return 0


def stage_md(args) -> int:
    from deepmd.infer import DeepPot

    out = BASE / args.run
    meta = json.loads((out / "labels.json").read_text())
    cheap, accurate = DeepPot(str(out / "r4" / "model.pth")), DeepPot(str(out / "r6" / "model.pth"))
    with h5py.File(TRAJ / f"{args.run}.h5", "r") as fh:
        numbers = np.asarray(fh.attrs["numbers"])
        cell = np.asarray(fh.attrs["cell"], dtype=np.float64)
        x0 = np.asarray(fh["positions"][args.start], dtype=np.float64)
        v0 = np.asarray(fh["velocities"][args.start], dtype=np.float64)  # ASE units
    from ase import units
    from ase.data import atomic_masses, chemical_symbols

    species = [chemical_symbols.index(s) for s in meta["type_map"]]
    atype = np.array([species.index(int(z)) for z in numbers])
    masses = atomic_masses[numbers]
    dt = args.dt * units.fs
    box = cell.reshape(1, 9)

    def force(model, x):
        e, f, _ = model.eval(x.reshape(1, -1), box, atype)
        return float(np.asarray(e).ravel()[0]), np.asarray(f).reshape(-1, 3)

    report = {"run": args.run, "start": args.start, "dt_fs": args.dt, "ps": args.ps, "every": args.every, "arms": {}}
    n_steps = int(round(args.ps * 1000.0 / args.dt))
    x_every = max(1, int(round(args.x_fs / args.dt)))
    for name in args.arms.split(","):
        pick = {"accurate": lambda i: accurate, "cheap": lambda i: cheap,
                "switching": lambda i: accurate if i % args.every == 0 else cheap}[name]
        x, v = x0.copy(), v0.copy()
        e, f = force(pick(0), x)
        e0 = e + 0.5 * float((masses[:, None] * v**2).sum())
        vs, xs = [], []  # the force call above compiled and warmed the model
        t0 = time.perf_counter()
        for i in range(n_steps):
            v = v + 0.5 * dt * f / masses[:, None]
            x = x + dt * v
            e, f = force(pick(i + 1), x)
            v = v + 0.5 * dt * f / masses[:, None]
            vs.append(v.astype(np.float32))
            if i % x_every == 0:
                xs.append(x.astype(np.float32))
        wall = time.perf_counter() - t0
        e1 = e + 0.5 * float((masses[:, None] * v**2).sum())
        row = {"ms_per_fs": 1e3 * wall / (n_steps * args.dt), "wall_s": wall,
               "energy_change_meV_per_atom": 1e3 * (e1 - e0) / len(numbers), "finite": bool(np.isfinite(x).all())}
        report["arms"][name] = row
        np.savez(RES / f"kanda_{args.run}_{args.tag}_{name}.npz", v=np.asarray(vs), x=np.asarray(xs), v_fs=args.dt,
                 x_fs=args.dt * x_every, numbers=numbers, cell=cell, masses=masses)
        print(f"{name:10s} {row['ms_per_fs']:8.3f} ms/fs | energy change {row['energy_change_meV_per_atom']:+.3f} "
              f"meV/atom over {args.ps:g} ps | finite {row['finite']}", flush=True)
    ref = report["arms"].get("accurate", {}).get("ms_per_fs")
    for row in report["arms"].values():
        row["speedup_vs_accurate"] = ref / row["ms_per_fs"] if ref else None
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"kanda_{args.run}_{args.tag}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: round(v["speedup_vs_accurate"] or 0, 2) for k, v in report["arms"].items()}), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("label", "config", "md"))
    ap.add_argument("--run", default="cu500_T300")
    ap.add_argument("--stride", type=int, default=4, help="label: every n-th trajectory frame")
    ap.add_argument("--steps", type=int, default=100000, help="config: DeePMD training steps (their example's)")
    ap.add_argument("--sel4", type=int, default=40)
    ap.add_argument("--sel6", type=int, default=120)
    ap.add_argument("--start", type=int, default=9000)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--ps", type=float, default=20.0)
    ap.add_argument("--every", type=int, default=4, help="md: the accurate model every n-th step (their 1:3)")
    ap.add_argument("--x-fs", type=float, default=64.0)
    ap.add_argument("--arms", default="accurate,switching,cheap")
    ap.add_argument("--tag", default="nve20")
    args = ap.parse_args()
    return {"label": stage_label, "config": stage_config, "md": stage_md}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
