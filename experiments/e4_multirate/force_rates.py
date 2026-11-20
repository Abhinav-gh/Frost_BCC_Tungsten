"""E4 pilot: are foundation-model forces slow once the fast part is removed?

Frost v1 froze tensor products inside the model and failed beyond MACE layer 1:
holding any input stale deletes the force path through it (plan 1.5-1.7). The
reframed design freezes one level up, at the model output. A cheap model runs
every step and the expensive model's residual

    R(t) = F_big(x_t) - F_small(x_t)

is refreshed only every k steps, as in r-RESPA multiple time stepping. That is
only worth doing if R changes much more slowly than F_big itself. This pilot
measures it on the real E1 trajectories (MACE-MP-0 medium, 1 fs, every step
stored), over every atom of every frame, never a single atom or edge. The
statistics (drift, hold, linear, lagged, energy acceptance proxy) are defined in
``frost.staleness``.

Each model is timed on a frame of the same cell, as a force call
(forward+backward) and as an energy-only forward, so cost ratios come from the
same GPU. MACE-MP-0 small has the same 6 A cutoff as medium and was not
distilled for this role, so its residual drift is an upper bound on what a
purpose-built short-range student would leave (E5 measures that student).

Output: data/results/e4/force_rates.json (forces eV/A, energies eV, times ms)

Usage (guarded launcher; GPU, float32)
  scripts/safe_run.sh e4_force_rates .venvs/frost-mace/bin/python \\
      experiments/e4_multirate/force_rates.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TRAJ = ROOT / "data" / "trajectories"
OUT = ROOT / "data" / "results" / "e4" / "force_rates.json"

RUNS = (
    "si512_T300", "silica1152_T300", "silica_glass1152_T300", "water216_T300",
    "cu500_T300", "li3po4_T300", "si512_T1000", "silica1152_T1000",
)
MODELS = ("small", "medium", "large")
PAIRS = (("medium", "small"), ("large", "medium"), ("large", "small"))
KS = (1, 2, 4, 8, 16, 32)


def main() -> int:
    from mace.calculators import mace_mp

    from frost.staleness import drift, energy_stats, hold, lagged, linear, rms
    from frost.teacher import frame_atoms, mace_forces, time_mace_ms
    from frost.trajectory import TrajectoryReader

    def timings(tag: str, model, batch, out: dict) -> None:
        for force, name in ((True, "ms"), (False, "energy_ms")):
            try:
                out[f"{tag}_{name}"] = time_mace_ms(model, batch, force=force)
            except Exception as exc:  # timing is secondary; drift is the result
                out[f"{tag}_{name}_error"] = f"{type(exc).__name__}: {exc}"[:200]

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--window", type=int, default=32, help="largest k; frames per window is k+1")
    ap.add_argument("--starts", default="1000,4000,7000")
    ap.add_argument("--only", default="", help="comma-separated subset of runs")
    ap.add_argument("--no-cueq", action="store_true", help="skip cuEquivariance timings")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    starts = [int(s) for s in args.starts.split(",")]
    runs = [r for r in RUNS if not args.only or r in args.only.split(",")]
    ks = [k for k in KS if k <= args.window]
    k4 = 4 if 4 in ks else ks[-1]
    calcs = {m: mace_mp(model=m, device=args.device, default_dtype="float32") for m in MODELS}

    report: dict = {"models": list(MODELS), "starts": starts, "window": args.window, "runs": {}}
    for run in runs:
        t0 = time.perf_counter()
        path = TRAJ / f"{run}.h5"
        if not path.exists():
            print(f"{run}: missing trajectory, skipped", flush=True)
            continue
        temperature = float(run.split("_T")[1])
        with TrajectoryReader(path) as reader:
            numbers = reader.numbers.copy()
            frames = [
                [frame_atoms(reader.numbers, reader.positions[s + j], reader.cell)
                 for j in range(args.window + 1)]
                for s in starts
            ]
        forces: dict[str, np.ndarray] = {}
        energies: dict[str, np.ndarray] = {}
        timing: dict[str, float | str] = {}
        n_edges = 0
        try:
            for m, calc in calcs.items():
                f_stack, e_stack = [], []
                batch = None
                for window in frames:
                    f_row, e_row = [], []
                    for atoms in window:
                        f, e, batch = mace_forces(calc, atoms, args.device)
                        f_row.append(f)
                        e_row.append(e)
                    f_stack.append(f_row)
                    e_stack.append(e_row)
                forces[m] = np.asarray(f_stack)  # (windows, frames, atoms, 3)
                energies[m] = np.asarray(e_stack)  # (windows, frames)
                n_edges = int(batch.edge_index.shape[1])
                timings(f"{m}_e3nn", calc.models[0], batch, timing)
                del batch
                torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"{run}: out of GPU memory, skipped", flush=True)
            torch.cuda.empty_cache()
            continue

        if not args.no_cueq:
            for m in MODELS:
                try:
                    fast = mace_mp(model=m, device=args.device, default_dtype="float32",
                                   enable_cueq=True)
                    _, _, batch = mace_forces(fast, frames[0][0], args.device)
                    timings(f"{m}_cueq", fast.models[0], batch, timing)
                    del fast, batch
                except Exception as exc:  # timing is optional; drift is the result
                    timing[f"{m}_cueq_error"] = f"{type(exc).__name__}: {exc}"[:200]
                torch.cuda.empty_cache()

        n_atoms = int(numbers.size)
        entry: dict = {
            "n_atoms": n_atoms,
            "n_edges": n_edges,
            "temperature_K": temperature,
            "timing": timing,
            "force_rms": {m: rms(forces[m]) for m in MODELS},
            "drift_F": {m: {k: drift(forces[m], k) for k in ks} for m in MODELS},
            "hold_F": {m: {k: hold(forces[m], k) for k in ks} for m in MODELS},
            "residual": {},
        }
        for big, small in PAIRS:
            r = forces[big] - forces[small]
            eres = energies[big] - energies[small]
            entry["residual"][f"{big}-{small}"] = {
                "level": rms(r),
                "drift_R": {k: drift(r, k) for k in ks},
                "hold_R": {k: hold(r, k) for k in ks},
                "lin_R": {k: linear(r, k) for k in ks},
                "lag1_R": {k: lagged(r, k) for k in ks},
                "energy": {k: energy_stats(eres, k, temperature, n_atoms) for k in ks},
            }
        r_ms = forces["medium"] - forces["small"]
        entry["by_species"] = {
            "k": k4,
            **{
                int(z): {
                    "drift_F_medium": drift(forces["medium"][:, :, numbers == z], k4),
                    "drift_R_medium-small": drift(r_ms[:, :, numbers == z], k4),
                }
                for z in np.unique(numbers)
            },
        }
        report["runs"][run] = entry

        res = entry["residual"]
        mev = 1e3  # JSON stays in eV/A; the log line is in meV/A
        k_acc = 16 if 16 in ks else ks[-1]
        print(
            f"{run}: N={n_atoms} | meV/A: dF_med k=1 "
            f"{entry['drift_F']['medium'][1] * mev:.0f}, k={k4} "
            f"{entry['drift_F']['medium'][k4] * mev:.0f} | dR k={k4} med-small "
            f"{res['medium-small']['drift_R'][k4] * mev:.0f}, large-med "
            f"{res['large-medium']['drift_R'][k4] * mev:.0f}"
            f" | acc(n_d={k_acc}) med-small "
            f"{res['medium-small']['energy'][k_acc]['acceptance_proxy']:.2f}"
            f" | force ms s/m/l {timing.get('small_e3nn_ms', float('nan')):.0f}/"
            f"{timing.get('medium_e3nn_ms', float('nan')):.0f}/"
            f"{timing.get('large_e3nn_ms', float('nan')):.0f}"
            f" [{time.perf_counter() - t0:.0f}s]",
            flush=True,
        )
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, indent=2))

    print(f"-> {OUT}")
    return 0 if report["runs"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
