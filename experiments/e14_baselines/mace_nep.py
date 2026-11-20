"""The MACE-NEP distillation baseline (Zhang et al., npj Comput. Mater. 12, 174, 2026), reproduced.

Their pipeline: a MACE model labels configurations of the target system, a NEP is distilled from
those labels, and the NEP alone runs the molecular dynamics in GPUMD. Ours, on our own systems and
teacher (MACE-MP-0 medium, the model Frost's runs use), in three stages and three environments,
because TorchNEP and GPUMD are GPL-3.0 tools used as separate programs and never imported by Frost:

1. ``labels`` (frost-mace): frames of the 300 K E1 trajectory every ``--stride`` fs over the first
   8 ps (training) and 8-10 ps (validation), labelled by MACE-MP-0 medium (energy, forces, virial),
   written as extended XYZ;
2. ``train`` (frost-nep): a NEP4 trained by TorchNEP on those labels (cutoffs 6 / 4 A, the NEP
   defaults GPUMD papers use);
3. ``md`` (frost-mace, calling the GPUMD binary): NVE dynamics with the NEP from the E1 frame Frost
   starts from, the speed GPUMD reports (atom-steps per second), and the trajectory scored with the
   same observables as ``validate.py`` against the teacher arm of a Frost fidelity run on the same
   system (``--teacher-ref``): RDF and VDOS against the teacher's own half-to-half spread.

Output: data/baselines/mace_nep/<run>/ (labels, nep.txt, GPUMD files) and
data/results/e14/mace_nep_<run>.json

Usage (guarded launcher)
  scripts/safe_run.sh mnep_lab .venvs/frost-mace/bin/python experiments/e14_baselines/mace_nep.py labels --run li3po4_T300
  scripts/safe_run.sh mnep_trn .venvs/frost-nep/bin/python experiments/e14_baselines/mace_nep.py train --run li3po4_T300
  scripts/safe_run.sh mnep_md  .venvs/frost-mace/bin/python experiments/e14_baselines/mace_nep.py md --run li3po4_T300 \\
      --teacher-ref data/results/e6/fidelity_li3po4_T300_nve_cheap_k6_50ps.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRAJ = ROOT / "data" / "trajectories"
BASE = ROOT / "data" / "baselines" / "mace_nep"
RES = ROOT / "data" / "results" / "e14"
GPUMD = ROOT / "third_party" / "GPUMD" / "src" / "gpumd"


def _write_xyz(path: Path, frames: list[dict]) -> None:
    with open(path, "w") as fh:
        for f in frames:
            lat = " ".join(f"{x:.10f}" for x in f["cell"].reshape(-1))
            props = "species:S:1:pos:R:3" + (":force:R:3" if "forces" in f else "") + (":vel:R:3" if "vel" in f else "")
            extra = ""
            if "energy" in f:
                extra += f' energy={f["energy"]:.10f}'
            if "virial" in f:
                extra += ' virial="' + " ".join(f"{x:.10f}" for x in f["virial"].reshape(-1)) + '"'
            fh.write(f"{len(f['symbols'])}\n")
            fh.write(f'Lattice="{lat}" Properties={props} pbc="T T T"{extra}\n')
            for i, s in enumerate(f["symbols"]):
                row = [f"{x:.10f}" for x in f["pos"][i]]
                if "forces" in f:
                    row += [f"{x:.10f}" for x in f["forces"][i]]
                if "vel" in f:
                    row += [f"{x:.10f}" for x in f["vel"][i]]
                fh.write(f"{s} " + " ".join(row) + "\n")


def work_dir(args) -> Path:
    """Labels and the trained NEP: data/baselines/mace_nep/<run>, or <run>_<tag> for a variant (``md --nep-from``)."""
    return BASE / (f"{args.run}_{args.tag}" if args.tag else args.run)


def labels(args) -> int:
    import h5py
    import torch
    from ase import Atoms
    from mace.calculators import mace_mp

    out = work_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(TRAJ / f"{args.run}.h5", "r") as fh:
        numbers = np.asarray(fh.attrs["numbers"])
        cell = np.asarray(fh.attrs["cell"], dtype=np.float64)
        pos = fh["positions"]
        n_frames = pos.shape[0]
        idx_train = list(range(0, min(8000, n_frames), args.stride))
        idx_valid = list(range(8000, min(10000, n_frames), args.stride))
        calc = mace_mp(model="medium", device="cuda", default_dtype="float64")
        for name, idx in (("train", idx_train), ("valid", idx_valid)):
            frames = []
            for i in idx:
                atoms = Atoms(numbers=numbers, positions=np.asarray(pos[i]), cell=cell, pbc=True)
                atoms.calc = calc
                e = atoms.get_potential_energy()
                f = atoms.get_forces()
                stress = atoms.get_stress(voigt=False)
                frames.append({"symbols": atoms.get_chemical_symbols(), "pos": atoms.positions, "cell": cell,
                               "energy": e, "forces": f, "virial": -stress * atoms.get_volume()})
            _write_xyz(out / f"{name}.xyz", frames)
            print(f"{name}: {len(frames)} frames labelled by MACE-MP-0 medium -> {out / f'{name}.xyz'}", flush=True)
        torch.cuda.empty_cache()
    return 0


def train(args) -> int:
    from torchnep import train_nep

    out = work_dir(args)
    symbols = []
    with open(out / "train.xyz") as fh:
        n = int(fh.readline())
        fh.readline()
        for _ in range(n):
            s = fh.readline().split()[0]
            if s not in symbols:
                symbols.append(s)
    cfg = out / "nep.in"
    cfg.write_text(f"type {len(symbols)} {' '.join(symbols)}\nversion 4\ncutoff 6 4\nn_max 6 6\nbasis_size 8 8\n"
                   f"l_max 4 2 0\nneuron {args.neuron}\n\nepoch {args.epochs}\nbatch 8\nlr 5e-3\n"
                   f"scheduler_factor 0.5\nstage2 1\nstart_stage2 {args.epochs // 2}\nstage2_lr 5e-4\n")
    t0 = time.perf_counter()
    train_nep(str(cfg), str(out / "train.xyz"), output_dir=str(out / "nep"), valid_file=str(out / "valid.xyz"))
    print(f"trained in {time.perf_counter() - t0:.0f} s -> {out / 'nep'}", flush=True)
    return 0


def md(args) -> int:
    import h5py
    import torch

    from frost.observables import rdf, vdos

    from ase import units

    out = BASE / args.run
    nep_src = BASE / (args.nep_from or args.run)
    nep_txt = next((nep_src / "nep").glob("nep_best.txt"), None) or next((nep_src / "nep").glob("nep*.txt"))
    run_dir = out / (f"gpumd_{args.tag}" if args.tag else "gpumd")
    run_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(TRAJ / f"{args.run}.h5", "r") as fh:
        numbers = np.asarray(fh.attrs["numbers"])
        cell = np.asarray(fh.attrs["cell"], dtype=np.float64)
        p0 = np.asarray(fh["positions"][args.start])
        v0 = np.asarray(fh["velocities"][args.start], dtype=np.float64) * units.fs  # ASE units -> A/fs
    if args.tile > 1:
        # a supercell of the frame, for GPUMD's speed at Frost's tiled size (timing only, as md_runtime --tile)
        offs = np.array([[a, b, c] for a in range(args.tile) for b in range(args.tile) for c in range(args.tile)]) @ cell
        p0 = np.concatenate([p0 + o for o in offs])
        v0 = np.concatenate([v0] * len(offs))
        numbers = np.concatenate([numbers] * len(offs))
        cell = cell * args.tile
    from ase import Atoms

    atoms = Atoms(numbers=numbers, positions=p0, cell=cell, pbc=True)
    frame = {"symbols": atoms.get_chemical_symbols(), "pos": p0, "cell": cell}
    if args.paired:
        frame["vel"] = v0  # the teacher arm's own starting velocities: a paired trajectory
    _write_xyz(run_dir / "model.xyz", [frame])
    text = Path(nep_txt).read_text()
    if args.max_neighbours:
        # GPUMD's neighbour buffers (the cutoff line's MN_radial MN_angular) are sizes, not model parameters; a
        # liquid or a hot start overflows the training-time ones and GPUMD faults with an illegal address
        lines = text.splitlines()
        i = next(n for n, line in enumerate(lines) if line.startswith("cutoff"))
        rc_r, rc_a = lines[i].split()[1:3]
        mn_r, mn_a = (int(v) for v in args.max_neighbours.split(":"))
        lines[i] = f"cutoff {rc_r} {rc_a} {mn_r} {mn_a}"
        text = "\n".join(lines) + "\n"
    (run_dir / "nep.txt").write_text(text)
    steps = round(args.fs / args.dt)
    every = max(1, round(2.0 / args.dt))
    start = "" if args.paired else (f"velocity {args.temperature}\n")
    warm = "" if args.paired else (f"ensemble nvt_ber {args.temperature} {args.temperature} 100\nrun 2000\n")  # 2 ps
    if args.ramp:
        t1, t2 = (float(v) for v in args.ramp.split(":"))
        ens = f"ensemble nvt_lan {t1} {t2} {1.0 / args.friction:.0f}\n"  # Langevin, time constant in fs
    else:
        ens = "ensemble nve\n"
    dump = "" if args.no_dump else f"dump_xyz {every} dump.xyz velocity\n"
    (run_dir / "run.in").write_text(f"potential nep.txt\n{start}time_step {args.dt}\n{warm}{ens}{dump}run {steps}\n")
    t0 = time.perf_counter()
    import os

    if args.rescore and (run_dir / "dump.xyz").exists():
        # score an existing GPUMD run against a (new) teacher reference without rerunning it
        class _Log:
            stdout = (run_dir / "gpumd.log").read_text()
            stderr = ""
            returncode = 0
        log = _Log()
        wall = float("nan")
    else:
        # one GPU: GPUMD otherwise decomposes the cell over every visible device, and these cells are smaller
        # than five cutoffs per domain (its own error message); the NEP's speed is then a one-GPU speed, as ours
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
        log = subprocess.run([str(GPUMD)], cwd=run_dir, capture_output=True, text=True, env=env)
        wall = time.perf_counter() - t0
        (run_dir / "gpumd.log").write_text(log.stdout + log.stderr)
    speeds = [float(x) for x in re.findall(r"Speed of this run = ([0-9.eE+-]+) atom\*step/second", log.stdout)]
    if log.returncode != 0 or not speeds:
        print(log.stdout[-3000:], log.stderr[-2000:], file=sys.stderr)
        raise SystemExit("GPUMD failed")
    if args.no_dump:
        report = {"run": args.run, "tag": args.tag, "tile": args.tile, "n_atoms": len(numbers), "nep": str(nep_txt),
                  "gpumd_atom_steps_per_s": speeds[-1], "gpumd_ms_per_step": 1e3 * len(numbers) / speeds[-1],
                  "fs": args.fs, "dt": args.dt}
        RES.mkdir(parents=True, exist_ok=True)
        (RES / f"mace_nep_{args.run}_{args.tag or 'timing'}.json").write_text(json.dumps(report, indent=1))
        print(json.dumps(report, indent=1), flush=True)
        return 0
    # the dumped NVE frames: positions and velocities (A, A/fs)
    xs, vs = [], []
    with open(run_dir / "dump.xyz") as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            n = int(line)
            fh.readline()
            rows = [fh.readline().split() for _ in range(n)]
            xs.append([[float(r[1]), float(r[2]), float(r[3])] for r in rows])
            vs.append([[float(r[4]), float(r[5]), float(r[6])] for r in rows])
    x = torch.as_tensor(np.asarray(xs), dtype=torch.float64, device="cuda")
    v = torch.as_tensor(np.asarray(vs), dtype=torch.float64, device="cuda")
    ref = json.loads(Path(args.teacher_ref).read_text())
    t = ref["teacher"]
    spread = ref.get("teacher_self") or {}
    _, types = np.unique(numbers, return_inverse=True)
    freq, spec, _ = vdos(v, every * args.dt, max_lag=1024)
    keep = freq <= 150.0
    g = rdf(x[::100].float(), cell, types, int(types.max()) + 1)
    def l1(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return float(np.abs(b - a).sum() / np.abs(a).sum())
    # the structure reference from the E1 trajectory with the fixed estimator (score's protocol): teacher arms of
    # fidelity files written before 2026-10-19 carry the unwrapped-frame RDF bug
    with h5py.File(TRAJ / f"{args.run}.h5", "r") as fh:
        xt = np.asarray(fh["positions"][4000:10000])
    def g_of(frames):
        return np.asarray(rdf(torch.as_tensor(frames, dtype=torch.float64), cell, types, int(types.max()) + 1)["g_total"])
    g_e1, g_e1a, g_e1b = g_of(xt[::200]), g_of(xt[:3000:200]), g_of(xt[3000::200])
    spec_n = np.interp(np.asarray(t["vdos_freq_THz"]), freq[keep], spec[keep] / spec[keep].max())
    report = {"run": args.run, "nep": str(nep_txt), "gpumd_atom_steps_per_s": speeds[-1],
              "gpumd_ms_per_step": 1e3 * len(numbers) / speeds[-1], "wall_s": wall, "fs": args.fs,
              "tag": args.tag, "paired": args.paired, "ensemble": args.ramp or "nve",
              "rdf_l1_vs_teacher": l1(g_e1, g["g_total"]),
              "rdf_teacher_own_spread": l1(g_e1a, g_e1b),
              "rdf_source": f"E1 trajectory {args.run}, frames 4000-10000 (fixed estimator)",
              "nep_vdos_first_moment_THz": float((freq[keep] * spec[keep]).sum() / spec[keep].sum()),
              "teacher_vdos_first_moment_THz": t.get("vdos_first_moment_THz"),
              "vdos_l1_vs_teacher": l1(t["vdos"], spec_n),
              "vdos_teacher_own_spread": spread.get("vdos_l1_relative"),
              "teacher_ms_per_fs": (ref["wall_seconds"]["teacher"] * 1e3 / ref["fs"]) if ref.get("wall_seconds") else None}
    # diffusion from velocities integrated over the dump interval (GPUMD's dumped positions are wrapped)
    dt_s = every * args.dt
    xu = torch.cumsum(0.5 * (v[1:] + v[:-1]) * dt_s, dim=0)
    lags = np.unique(np.linspace(1, xu.shape[0] // 2, 40).astype(int))
    msd_ = np.array([float(((xu[lag:] - xu[:-lag]) ** 2).sum(-1).mean()) for lag in lags])
    t_ps = lags * dt_s / 1e3
    sel = t_ps >= 0.2 * t_ps[-1]
    report["diffusion_A2_per_ps"] = float(np.polyfit(t_ps[sel], msd_[sel], 1)[0] / 6.0)
    report["teacher_diffusion_A2_per_ps"] = t.get("diffusion_A2_per_ps")
    if report["teacher_ms_per_fs"]:
        report["speedup_vs_teacher"] = report["teacher_ms_per_fs"] / report["gpumd_ms_per_step"] * args.dt
    RES.mkdir(parents=True, exist_ok=True)
    (RES / (f"mace_nep_{args.run}_{args.tag}.json" if args.tag else f"mace_nep_{args.run}.json")).write_text(
        json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items()}, indent=1), flush=True)
    return 0


def _read_dump(path: Path, every: int = 1):
    xs, vs = [], []
    with open(path) as fh:
        i = 0
        while True:
            line = fh.readline()
            if not line:
                break
            n = int(line)
            fh.readline()
            rows = [fh.readline().split() for _ in range(n)]
            if i % every == 0:
                xs.append([[float(r[1]), float(r[2]), float(r[3])] for r in rows])
                vs.append([[float(r[4]), float(r[5]), float(r[6])] for r in rows])
            i += 1
    return np.asarray(xs), np.asarray(vs)


def score(args) -> int:
    """The NEP's trajectory against the teacher's own: the E1 MACE-MP-0 medium trajectory of the same system and
    temperature (1 fs frames), both scored with the same estimators, plus the teacher's half-to-half spread. This
    replaces ``md``'s comparison against a fidelity file's teacher arm, whose RDF predates the fix of
    ``observables.rdf`` for unwrapped frames (2026-10-19)."""
    import h5py
    import torch

    from frost.observables import rdf, vdos

    out = BASE / args.run
    with h5py.File(TRAJ / f"{args.run}.h5", "r") as fh:
        numbers = np.asarray(fh.attrs["numbers"])
        cell = np.asarray(fh.attrs["cell"], dtype=np.float64)
        xt = np.asarray(fh["positions"][4000:10000])
        vt = np.asarray(fh["velocities"][4000:10000])
    _, types = np.unique(numbers, return_inverse=True)
    nt = int(types.max()) + 1
    xn, vn = _read_dump(out / "gpumd" / "dump.xyz")  # every 2 fs
    def g(x):
        return np.asarray(rdf(torch.as_tensor(x, dtype=torch.float64), cell, types, nt)["g_total"])
    def spec(v, dt_fs):
        f, s_, _ = vdos(torch.as_tensor(v, dtype=torch.float64, device="cuda"), dt_fs, max_lag=1024)
        keep = f <= 60.0
        return f[keep], s_[keep] / s_[keep].max()
    def l1(a, b):
        return float(np.abs(b - a).sum() / np.abs(a).sum())
    gt, gt1, gt2 = g(xt[::200]), g(xt[:3000:200]), g(xt[3000::200])
    gn = g(xn[::100])
    # the spectrum from a live NVE teacher arm: the E1 trajectories were thermostatted, and Langevin noise
    # flattens a velocity spectrum (Cu's first moment reads 6.76 THz there against 4.74 in NVE)
    ref = json.loads(Path(args.teacher_ref).read_text())
    ft, st = np.asarray(ref["teacher"]["vdos_freq_fine_THz"]), np.asarray(ref["teacher"]["vdos_fine"])
    fn, sn = spec(vn, 2.0)
    sn_i = np.interp(ft, fn, sn) * (st.max() / max(np.interp(ft, fn, sn).max(), 1e-30))
    own_vdos = (ref.get("teacher_self") or ref["teacher"].get("self") or {}).get("vdos_l1_relative")
    report = json.loads((RES / f"mace_nep_{args.run}.json").read_text()) if (RES / f"mace_nep_{args.run}.json").exists() else {}
    report.update({"teacher_source": f"E1 trajectory {args.run}, frames 4000-10000", "rdf_l1_vs_teacher": l1(gt, gn),
                   "rdf_teacher_own_spread": l1(gt1, gt2), "vdos_l1_vs_teacher": l1(st, sn_i),
                   "vdos_teacher_own_spread": own_vdos, "vdos_teacher_source": args.teacher_ref,
                   "nep_vdos_first_moment_THz": float((fn * sn).sum() / sn.sum()),
                   "teacher_vdos_first_moment_THz": ref["teacher"].get("vdos_first_moment_THz"), "rdf_fixed": True})
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"mace_nep_{args.run}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("rdf_l1_vs_teacher", "rdf_teacher_own_spread", "vdos_l1_vs_teacher",
                                             "vdos_teacher_own_spread", "nep_vdos_first_moment_THz",
                                             "teacher_vdos_first_moment_THz")}, indent=1), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("labels", "train", "md", "score"))
    ap.add_argument("--run", required=True)
    ap.add_argument("--stride", type=int, default=16, help="fs between labelled frames")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--neuron", type=int, default=30)
    ap.add_argument("--start", type=int, default=9000)
    ap.add_argument("--fs", type=int, default=10000)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--teacher-ref", default="")
    ap.add_argument("--paired", action="store_true",
                    help="md: start from the frame's own velocities (no random start, no NVT warm-up)")
    ap.add_argument("--tile", type=int, default=1, help="md: supercell of the frame (timing only)")
    ap.add_argument("--no-dump", action="store_true", help="md: timing only, no trajectory output")
    ap.add_argument("--ramp", default="", help="md: T1:T2, Langevin with a linear ramp of the target")
    ap.add_argument("--friction", type=float, default=0.01, help="md --ramp: per fs")
    ap.add_argument("--nep-from", default="", help="md: use the NEP trained on this run (e.g. cu500_T300)")
    ap.add_argument("--rescore", action="store_true", help="md: reuse an existing dump.xyz (no GPUMD run)")
    ap.add_argument("--max-neighbours", default="", help="md: MN_radial:MN_angular for GPUMD's buffers (e.g. 200:80)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    return {"labels": labels, "train": train, "md": md, "score": score}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
