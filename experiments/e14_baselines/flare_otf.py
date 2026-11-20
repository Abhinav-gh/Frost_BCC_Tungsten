"""FLARE on-the-fly learning (Vandermause et al., npj Comput. Mater. 2020 and 2022) on our system, MACE as its oracle (T1).

FLARE's sparse Gaussian process predicts forces with an uncertainty at every step and calls its oracle (DFT in their
papers) whenever the uncertainty passes a threshold, adding the uncertain environments to its training set. That
uncertainty gate is the closest published analogue of Frost's certificate, so the comparison asks two things: how
many oracle calls each needs per ps, and whether FLARE's gate keeps the error of the steps it accepts small.

FLARE runs unmodified from its clone (``third_party/flare``, built into ``.venvs/frost-flare``) through its public
pieces: ``flare.scripts.otf_train.get_sgp_calc`` builds the sparse GP from a config dict, ``flare.learners.otf.OTF``
runs the learning MD, ``flare.io.otf_parser.OtfAnalysis`` reads its output. The oracle is our MACE teacher served from
the frost-mace environment over pipes (``frost.external_teacher``, ``mace:<teacher>``), wrapped as an ASE calculator,
in place of their DFT calculator: the one change to their protocol. Settings are their ``examples/full.yaml``'s
(B2, nmax 8, lmax 4, 5 A, normalised dot product kernel of power 2, local uncertainty, absolute threshold 0.02,
threshold updates at 0.01, energy, force and stress training, hyperparameters trained at every call), with the single
atom energy set to the oracle's mean energy per atom at the start frame (their tutorial uses isolated-atom energies).

Arms, from the same start frame of one of our trajectories:
- ``reference``: the oracle every step through ASE's velocity Verlet (FLARE's "DFT" MD), the fidelity reference;
- ``flare``: FLARE's OTF with ASE's velocity Verlet (their ASE engine; their fast path, LAMMPS with the mapped SGP,
  is not built here, so its wall time is FLARE's Python and C++ on 4 CPU threads).

After the run the oracle is called on every ``--check-every``-th accepted (model) step, outside the timed region, to
measure the force error of the steps FLARE trusted. Frames go to ``data/results/e14/flare_<run>_<tag>_<arm>.npz`` in
``flashmd_run.py``'s layout, for ``flashmd_score.py <run> <tag> flare``.

Usage (guarded launcher; the oracle on GPU)
  source scripts/env.sh mace; LANE=gpu1 scripts/safe_run.sh flare_cu env PYTHONPATH=src \\
      .venvs/frost-flare/bin/python experiments/e14_baselines/flare_otf.py --run cu500_T300 --ps 10 --tag nve10
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import ase
import ase.units as units
import h5py
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.md.verlet import VelocityVerlet

from frost.external_teacher import ExternalTeacher

ROOT = Path(__file__).resolve().parents[2]
TRAJ = ROOT / "data" / "trajectories"
RES = ROOT / "data" / "results" / "e14"
WORK = ROOT / "data" / "baselines" / "flare"


class Oracle(Calculator):
    """An ASE calculator over an ``ExternalTeacher`` (energy, forces and stress)."""

    implemented_properties = ("energy", "free_energy", "forces", "stress")

    def __init__(self, teacher: ExternalTeacher):
        super().__init__()
        self.teacher = teacher

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        e, f = self.teacher.numpy(self.atoms.get_positions())
        self.results = {"energy": e, "free_energy": e, "forces": np.array(f), "stress": self.teacher.last_stress.copy()}


def start_atoms(run: str, frame: int) -> ase.Atoms:
    with h5py.File(TRAJ / f"{run}.h5", "r") as fh:
        numbers = np.asarray(fh.attrs["numbers"])
        cell = np.asarray(fh.attrs["cell"], dtype=np.float64)
        x = np.asarray(fh["positions"][frame], dtype=np.float64)
        v = np.asarray(fh["velocities"][frame], dtype=np.float64)  # ASE units, as our trajectories store them
    atoms = ase.Atoms(numbers=numbers, positions=x, cell=cell, pbc=True)
    atoms.set_velocities(v)
    return atoms


def write_report(args, report):
    """Merge into the report on disk (another job may have added an arm since this one started), atomically."""
    path = RES / f"flare_{args.run}_{args.tag}.json"
    if path.exists():
        report["arms"] = {**json.loads(path.read_text())["arms"], **report["arms"]}
    ref = report["arms"].get("reference", {}).get("ms_per_fs")
    for row in report["arms"].values():
        row["speedup_vs_reference"] = ref / row["ms_per_fs"] if ref else None
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(report, indent=1))
    os.replace(tmp, path)


def save_arm(name, args, atoms, vs, xs, x_every, row, report):
    report["arms"][name] = row
    np.savez(RES / f"flare_{args.run}_{args.tag}_{name}.npz", v=np.asarray(vs, dtype=np.float32),
             x=np.asarray(xs, dtype=np.float32), v_fs=args.dt, x_fs=args.dt * x_every,
             numbers=atoms.get_atomic_numbers(), cell=np.asarray(atoms.get_cell()), masses=atoms.get_masses())
    write_report(args, report)


def run_reference(args, oracle, n_steps, x_every, report):
    atoms = start_atoms(args.run, args.start)
    atoms.calc = oracle
    dyn = VelocityVerlet(atoms, timestep=args.dt * units.fs)
    e0 = atoms.get_potential_energy() + atoms.get_kinetic_energy()
    calls0, secs0 = oracle.teacher.calls, oracle.teacher.seconds
    vs, xs = [], []
    t0 = time.perf_counter()
    for i in range(n_steps):
        dyn.run(1)
        vs.append(atoms.get_velocities().astype(np.float32))
        if i % x_every == 0:
            xs.append(atoms.get_positions().astype(np.float32))
    wall = time.perf_counter() - t0
    e1 = atoms.get_potential_energy() + atoms.get_kinetic_energy()
    calls = oracle.teacher.calls - calls0
    row = {"wall_s": wall, "ms_per_fs": 1e3 * wall / (n_steps * args.dt), "oracle_calls": calls,
           "oracle_ms_per_call": 1e3 * (oracle.teacher.seconds - secs0) / max(calls, 1),
           "energy_change_meV_per_atom": 1e3 * (e1 - e0) / len(atoms), "finite": bool(np.isfinite(xs).all())}
    save_arm("reference", args, atoms, vs, xs, x_every, row, report)
    print(f"reference  {row['ms_per_fs']:8.3f} ms/fs | oracle {row['oracle_ms_per_call']:.2f} ms/call | energy change "
          f"{row['energy_change_meV_per_atom']:+.3f} meV/atom | finite {row['finite']}", flush=True)


def run_flare(args, oracle, n_steps, x_every, report):
    from flare.io.otf_parser import OtfAnalysis
    from flare.learners.otf import OTF
    from flare.scripts.otf_train import get_sgp_calc

    atoms = start_atoms(args.run, args.start)
    species = sorted({int(z) for z in atoms.numbers})
    probe = atoms.copy()
    probe.calc = oracle
    e_start = probe.get_potential_energy()
    by_species = {z: int((atoms.numbers == z).sum()) for z in species}
    sae = [e_start / len(atoms)] * len(species)  # one species here; several would need per-species references
    config = {"kernels": [{"name": "NormalizedDotProduct", "sigma": 2.0, "power": 2}],
              "descriptors": [{"name": "B2", "nmax": args.nmax, "lmax": args.lmax, "cutoff_function": "quadratic",
                               "radial_basis": "chebyshev"}],
              "energy_noise": 0.01, "forces_noise": 0.05, "stress_noise": 0.005, "species": species,
              "single_atom_energies": sae, "cutoff": args.cutoff, "variance_type": "local", "max_iterations": 1,
              "use_mapping": False}
    flare_calc, _ = get_sgp_calc(config)
    work = WORK / f"{args.run}_{args.tag}"
    work.mkdir(parents=True, exist_ok=True)
    out_name = str(work / "otf")
    calls0, secs0 = oracle.teacher.calls, oracle.teacher.seconds
    otf = OTF(atoms, dt=args.dt * 1e-3, number_of_steps=n_steps, dft_calc=oracle, md_engine="VelocityVerlet",
              md_kwargs={}, flare_calc=flare_calc, output_name=out_name, std_tolerance_factor=args.std_tolerance,
              max_atoms_added=-1, train_hyps=(0, np.inf), update_style="threshold",
              update_threshold=args.update_threshold, write_model=1, force_only=False,
              init_atoms=list(range(args.init_atoms)), skip=0)
    t0 = time.perf_counter()
    otf.run()
    wall = time.perf_counter() - t0
    calls = oracle.teacher.calls - calls0
    oracle_s = oracle.teacher.seconds - secs0

    log = OtfAnalysis(out_name + ".out")
    # one record per step: model frames ("-Frame") and, at oracle steps, the frame after the oracle call
    frames = {}
    for t, x, v, f, s in zip(log.times, log.position_list, log.velocity_list, log.force_list, log.uncertainty_list,
                                  strict=True):
        frames[round(t * 1e3 / args.dt)] = (x, v, f, s, False)
    dft_steps = sorted(set(log.dft_frames))
    for k, step in enumerate(log.dft_frames):  # two records per call (model, then oracle); keep the later one
        frames[step] = (log.gp_position_list[k], log.gp_velocity_list[k], None, None, True)
    steps = sorted(s for s in frames if 0 <= s < n_steps)
    to_ase = 1.0 / (1e3 * units.fs)  # FLARE prints velocities in A/ps
    vs = [np.asarray(frames[s][1]) * to_ase for s in steps]
    xs = [np.asarray(frames[s][0]) for s in steps[::x_every]]
    e1 = None
    if len(steps) != n_steps:
        print(f"warning: {len(steps)} recorded steps for {n_steps}", flush=True)

    # the steps FLARE trusted: oracle force against FLARE's, outside the timed region
    checks = []
    accepted = [s for s in steps if not frames[s][4] and frames[s][2] is not None]
    for s in accepted[:: args.check_every]:
        x, _, f_model, std, _ = frames[s]
        _, f_true = oracle.teacher.numpy(np.asarray(x, dtype=np.float64))
        err = np.linalg.norm(np.asarray(f_model) - f_true, axis=1)
        checks.append({"step": s, "max_err": float(err.max()), "mean_err": float(err.mean()),
                       "rms_err": float(np.sqrt((err**2).mean())), "max_std": float(np.abs(std).max())})
    row = {"wall_s": wall, "ms_per_fs": 1e3 * wall / (n_steps * args.dt), "oracle_calls": calls,
           "oracle_calls_per_ps": calls / (n_steps * args.dt * 1e-3), "oracle_wall_s": oracle_s,
           "flare_only_ms_per_fs": 1e3 * (wall - oracle_s) / (n_steps * args.dt), "dft_steps": dft_steps,
           "sparse_size": int(otf.gp.sparse_gp.Kuu.shape[0]) if hasattr(otf.gp.sparse_gp, "Kuu") else None,
           "recorded_steps": len(steps), "accepted_checks": checks, "settings": config,
           "std_tolerance": args.std_tolerance, "update_threshold": args.update_threshold,
           "init_atoms": args.init_atoms, "species_counts": by_species,
           "energy_change_meV_per_atom": e1, "finite": bool(np.isfinite(xs).all())}
    if checks:
        errs = np.array([c["max_err"] for c in checks])
        row["accepted_max_err_eV_A"] = {"median": float(np.median(errs)), "p95": float(np.percentile(errs, 95)),
                                       "max": float(errs.max())}
        row["accepted_rms_err_eV_A"] = float(np.sqrt(np.mean([c["rms_err"] ** 2 for c in checks])))
    save_arm("flare", args, atoms, vs, xs, x_every, row, report)
    print(f"flare      {row['ms_per_fs']:8.3f} ms/fs ({row['flare_only_ms_per_fs']:.3f} without the oracle) | "
          f"{calls} oracle calls ({row['oracle_calls_per_ps']:.1f} per ps) | accepted-step force error "
          f"{row.get('accepted_max_err_eV_A')} max per atom, rms {row.get('accepted_rms_err_eV_A')} eV/A", flush=True)
    if args.cleanup:
        for suffix in (".out", ".xyz"):  # the per-step text logs, now in the .npz
            Path(out_name + suffix).unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="cu500_T300")
    ap.add_argument("--start", type=int, default=9000)
    ap.add_argument("--oracle", default="mace:mp-medium")
    ap.add_argument("--dt", type=float, default=1.0, help="fs")
    ap.add_argument("--ps", type=float, default=10.0)
    ap.add_argument("--x-fs", type=float, default=64.0)
    ap.add_argument("--arms", default="reference,flare")
    ap.add_argument("--cutoff", type=float, default=5.0)
    ap.add_argument("--nmax", type=int, default=8)
    ap.add_argument("--lmax", type=int, default=4)
    ap.add_argument("--std-tolerance", type=float, default=-0.02, help="their example's absolute threshold")
    ap.add_argument("--update-threshold", type=float, default=0.01)
    ap.add_argument("--init-atoms", type=int, default=12, help="environments from the first frame (their example's 12)")
    ap.add_argument("--check-every", type=int, default=50, help="oracle check on every n-th accepted step")
    ap.add_argument("--cleanup", action="store_true", help="delete FLARE's per-step text log after parsing")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    RES.mkdir(parents=True, exist_ok=True)
    atoms = start_atoms(args.run, args.start)
    oracle = Oracle(ExternalTeacher(args.oracle, atoms.numbers, np.asarray(atoms.get_cell()), stress=True))
    n_steps = round(args.ps * 1000.0 / args.dt)
    x_every = max(1, round(args.x_fs / args.dt))
    report = {"run": args.run, "start": args.start, "oracle": args.oracle, "ps": args.ps, "dt_fs": args.dt, "arms": {}}
    prior = RES / f"flare_{args.run}_{args.tag}.json"
    if prior.exists():  # arms run as separate jobs (CPU lane, GPU lane) share one report: same settings only
        old = json.loads(prior.read_text())
        if (old["run"], old["start"], old["oracle"], old["ps"], old["dt_fs"]) != (args.run, args.start, args.oracle,
                                                                                 args.ps, args.dt):
            raise SystemExit(f"{prior} was written with other settings; pass another --tag")
    try:
        for arm in args.arms.split(","):
            {"reference": run_reference, "flare": run_flare}[arm](args, oracle, n_steps, x_every, report)
    finally:
        oracle.teacher.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
