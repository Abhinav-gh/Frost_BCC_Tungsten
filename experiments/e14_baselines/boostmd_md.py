"""BoostMD (our validated rebuild, ``experiments/e14_baselines/boostmd.py``) run as an MD arm beside Frost's own runs.

    python experiments/e14_baselines/boostmd_md.py --run ala2gas_T300 --start 9000 --fs 50000 --ref-every 10 \
        --frames-like data/results/e13/frames_ala2gas_T300_<tag>.npz --tag <tag>

The comparison's protocol (fair-comparison skill; spec in ``boostmd_spec.md``, "The comparison after validation"): the
same start frame as Frost's ``md_runtime.py`` run (positions and velocities of frame --start of the run's trajectory),
the same integrator and thermostat (``frost.integrators``: BAOAB, Langevin with the same friction, temperature and seed),
the same step, and frames recorded the same way (``md_runtime.Frames``), so ``score_frames.py`` and ``dihedrals.py``
grade BoostMD and Frost against the same teacher arm. The system is the run's copies of 22 atoms
(``frost.systems.ala2_gas``); edges stay within a copy, as in BoostMD's own batching (copies never meet).

Each step's force: every --ref-every steps the reference model (MACE-OFF23-M, float32, cuEquivariance) gives forces,
per-atom energies and the scalar node features the BoostMD model reads until the next reference (their Fig. 3, N = 10);
in between, the BoostMD model's forces. Timing: wall time between device syncs around the whole loop, frames kept on the
GPU (no sync inside); BoostMD's force error against the reference (their criterion: Table 1's errors some steps after a
reference) is sampled every --err-every steps with an extra reference call whose GPU time is measured with CUDA events and
left out. Component costs (one reference call, one BoostMD call, eager) are timed at the start frame.

Outputs: ``frames_<run>_boostmd<N>_<tag>.npz`` (the teacher arm copied from --frames-like after checking atoms, masses
and strides agree, and the BoostMD arm as ``boostmd<N>``) and ``boostmd_md_<run>_n<N>_<tag>.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from ase import Atoms, units

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "e14_baselines"))
sys.path.insert(0, str(ROOT / "experiments" / "e13_runtime"))

import boostmd as bm  # noqa: E402  the validated rebuild: model, loader, prediction
from md_runtime import Frames  # noqa: E402  the frame recorder Frost's arms use

from frost.integrators import Integrator, Langevin, MDState  # noqa: E402
from frost.teachers import load_calc  # noqa: E402

OUT = ROOT / "data" / "results" / "e13"
COPY = 22  # atoms per alanine dipeptide copy (frost.systems.ala2_gas)


class Reference:
    """MACE-OFF23-M on the copies: per-atom energies, forces and l = 0 node features, edges within each copy."""

    def __init__(self, atoms: Atoms, dev: str):
        from frost.calibrate import _batch_for

        calc = load_calc("off-medium", dev, "float32", enable_cueq=True)
        self.model = calc.models[0]
        self.data = _batch_for(calc, atoms, bm.R_MAX, dev)[0].to_dict()
        self.dtype = self.data["positions"].dtype
        n = len(atoms)
        if n % COPY:
            raise SystemExit(f"{n} atoms is not a whole number of {COPY}-atom copies")
        a = np.arange(COPY)
        i, j = np.meshgrid(a, a, indexing="ij")
        m = i != j
        offs = np.arange(0, n, COPY)[:, None]
        self.ci = torch.as_tensor((i[m][None] + offs).ravel(), device=dev)
        self.cj = torch.as_tensor((j[m][None] + offs).ravel(), device=dev)
        self.scal = bm.scalar_index(self.model).to(dev)

    def edges(self, x: torch.Tensor):
        keep = (x[self.cj] - x[self.ci]).norm(dim=1) < bm.R_MAX
        return self.ci[keep], self.cj[keep]

    def __call__(self, x: torch.Tensor):
        ci, cj = self.edges(x)
        n_e = ci.numel()
        self.data["edge_index"] = torch.stack((ci, cj))
        self.data["shifts"] = torch.zeros(n_e, 3, dtype=self.dtype, device=x.device)
        self.data["unit_shifts"] = torch.zeros(n_e, 3, dtype=self.dtype, device=x.device)
        self.data["positions"] = x.detach().to(self.dtype).clone()
        out = self.model(self.data, compute_force=True, training=False)
        return (out["node_energy"].detach().double(), out["forces"].detach().double(),
                out["node_feats"].detach()[:, self.scal].float())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--start", type=int, default=9000)
    ap.add_argument("--fs", type=float, default=50000.0)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--friction", type=float, default=0.001, help="per fs (md_runtime's NVT default)")
    ap.add_argument("--temp", type=float, default=0.0, help="K; 0: the trajectory's")
    ap.add_argument("--seed", type=int, default=0, help="the Langevin seed (md_runtime --seed)")
    ap.add_argument("--ref-every", type=int, default=10, help="N: steps between reference calls (their Fig. 3: 10)")
    ap.add_argument("--err-every", type=int, default=500, help="steps between untimed force-error samples")
    ap.add_argument("--base-fs", type=float, default=2000.0, help="the reference every step, same process (baseline)")
    ap.add_argument("--frames-like", required=True, help="md_runtime frames file of the same run and start")
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    if args.err_every % args.ref_every:
        raise SystemExit("--err-every must be a multiple of --ref-every")
    dev = "cuda"
    torch.manual_seed(args.seed)
    with h5py.File(ROOT / "data" / "trajectories" / f"{args.run}.h5", "r") as fh:
        numbers = np.asarray(fh.attrs["numbers"])
        cell = np.asarray(fh.attrs["cell"], dtype=np.float64)
        x0 = np.asarray(fh["positions"][args.start], dtype=np.float64)
        v0 = np.asarray(fh["velocities"][args.start], dtype=np.float64)
        temp = args.temp or float(fh.attrs["temperature_K"])
    atoms = Atoms(numbers=numbers, positions=x0, cell=cell, pbc=True)
    like = np.load(args.frames_like)
    if not np.array_equal(like["numbers"], numbers):
        raise SystemExit("--frames-like holds other atoms")
    ref = Reference(atoms, dev)
    model = bm.load_trained("boost", dev, args.ckpt)
    z = torch.as_tensor([bm.SPECIES.index(int(q)) for q in numbers], device=dev)
    attrs = torch.nn.functional.one_hot(z, len(bm.SPECIES)).float()
    batch = torch.as_tensor(np.arange(len(numbers)) // COPY, device=dev)
    n_mol = len(numbers) // COPY
    masses = torch.as_tensor(atoms.get_masses(), dtype=torch.float64, device=dev).unsqueeze(1)
    if not np.allclose(like["masses"], atoms.get_masses()):
        raise SystemExit("--frames-like has other masses (HMR?): BoostMD runs the trajectory's masses")
    steps = round(args.fs / args.dt)

    def boost_force(x, held):
        ci, cj = ref.edges(x)
        b = {"x": x.float(), "x_ref": held[0].float(), "h_ref": held[1], "e_ref": held[2].float(), "attrs": attrs,
             "edge_index": torch.stack((cj, ci)), "batch": batch, "n_graphs": n_mol}
        e, f = bm.predict("boost", model, b, create_graph=False)
        return e.detach().sum().double(), f.detach().double()

    # component costs at the start frame (eager, as run): one reference call, one BoostMD call
    xs = torch.as_tensor(x0, dtype=torch.float64, device=dev)
    e_at, f_ref0, feats = ref(xs)
    held0 = (xs.clone(), feats, e_at)

    def cost(fn, reps=20):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return 1e3 * (time.perf_counter() - t0) / reps

    c_ref = cost(lambda: ref(xs))
    c_boost = cost(lambda: boost_force(xs, held0))
    print(f"{args.run}: {len(numbers)} atoms, {n_mol} copies, T {temp:.0f} K; reference call {c_ref:.2f} ms, BoostMD call "
          f"{c_boost:.2f} ms ({c_ref / c_boost:.2f}x cheaper)", flush=True)

    def run(n_steps, every, record, err_every):
        state = MDState(torch.as_tensor(x0, dtype=torch.float64, device=dev).clone(),
                        torch.as_tensor(v0, dtype=torch.float64, device=dev).clone(), masses)
        integ = Integrator(state, args.dt, thermostat=Langevin(args.dt, args.friction, temp, masses, seed=args.seed))
        frames = Frames(args.dt, v_fs=float(like["v_fs"]), x_fs=float(like["x_fs"])) if record else None
        box = {"step": 0, "held": None, "ref_calls": 0, "boost_calls": 0}
        errs, untimed, temps = {}, [], []

        def force(x):
            s = box["step"]
            if s % every == 0:
                e_at, f, feats = ref(x)
                box["held"] = (x.detach().clone(), feats, e_at)
                box["ref_calls"] += 1
                return e_at.sum(), f, x
            e, f = boost_force(x, box["held"])
            box["boost_calls"] += 1
            # one sample per block of err_every steps, the offset since the reference cycling through 1 .. N - 1
            # (err_every is a multiple of N, so s % N equals the position inside the block)
            if err_every and every > 1 and s % err_every == 1 + (s // err_every) % (every - 1):
                a, z_ = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                _, f_true, _ = ref(x)
                d = f - f_true
                errs.setdefault(s % every, []).append(torch.stack([(d * d).mean().sqrt(), (f_true * f_true).mean().sqrt()]))
                z_.record()
                untimed.append((a, z_))
            return e, f, x

        f = force(state.x)[1]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(n_steps):
            box["step"] = i + 1
            _, f = integ.inner_step(force, f)
            if frames is not None:
                frames(i + 1, state)
            if (i + 1) % 1000 == 0:
                temps.append(state.temperature())
        torch.cuda.synchronize()
        sec = time.perf_counter() - t0 - 1e-3 * sum(a.elapsed_time(z_) for a, z_ in untimed)
        err_rows = {int(o): (1e3 * torch.stack(v).mean(0)).tolist() for o, v in sorted(errs.items())}
        return sec, frames, box, err_rows, temps, bool(torch.isfinite(state.x).all())

    sec_b, _, _, _, temps_b, _ = run(round(args.base_fs / args.dt), 1, False, 0)
    base_ms = 1e3 * sec_b / args.base_fs
    print(f"baseline (the reference every step, this process): {base_ms:.2f} ms/fs", flush=True)
    sec, frames, box, err_rows, temps, finite = run(steps, args.ref_every, True, args.err_every)
    ms = 1e3 * sec / args.fs
    name = f"boostmd{args.ref_every}"
    res = {"run": args.run, "start": args.start, "fs": args.fs, "ref_every": args.ref_every, "friction_per_fs": args.friction,
           "seed": args.seed, "temperature_K": temp, "atoms": len(numbers), "copies": n_mol,
           "cost_ms": {"reference_call": c_ref, "boost_call": c_boost}, "baseline_ms_per_fs": base_ms,
           "ms_per_fs": ms, "speedup_vs_same_process_baseline": base_ms / ms, "reference_calls": box["ref_calls"],
           "boost_calls": box["boost_calls"], "finite": finite, "T_mean": float(np.mean(temps)) if temps else None,
           "T_sd": float(np.std(temps)) if temps else None, "baseline_T_mean": float(np.mean(temps_b)) if temps_b else None,
           "force_error_meV_per_A_by_offset": {o: {"rmse": r[0], "rms_force": r[1]} for o, r in err_rows.items()},
           "gpu": torch.cuda.get_device_name(0)}
    (OUT / f"boostmd_md_{args.run}_n{args.ref_every}_{args.tag}.json").write_text(json.dumps(res, indent=1))
    v, x = frames.arrays()
    for kind, arr in (("v", v), ("x", x)):
        if arr.shape[0] != like[f"teacher__{kind}"].shape[0]:
            raise SystemExit(f"{kind} frames {arr.shape[0]} against the teacher's {like[f'teacher__{kind}'].shape[0]}")
    merged = {k: like[k] for k in ("numbers", "cell", "masses", "v_fs", "x_fs")}
    merged["teacher__v"], merged["teacher__x"] = like["teacher__v"], like["teacher__x"]
    merged[f"{name}__v"], merged[f"{name}__x"] = v, x
    out = OUT / f"frames_{args.run}_{name}_{args.tag}.npz"
    np.savez(out, **merged)
    print(f"{name}: {ms:.2f} ms/fs ({base_ms / ms:.2f}x over this process's baseline) | reference calls {box['ref_calls']}, "
          f"BoostMD calls {box['boost_calls']} | T {res['T_mean'] or float('nan'):.1f} K "
          f"(baseline {res['baseline_T_mean'] or float('nan'):.1f}) | "
          f"force error by steps since the reference (meV/A): "
          + ", ".join(f"{o}: {r['rmse']:.0f}" for o, r in res["force_error_meV_per_A_by_offset"].items()), flush=True)
    print("->", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
