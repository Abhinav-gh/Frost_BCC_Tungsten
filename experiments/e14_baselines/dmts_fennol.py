"""DMTS rerun on our GPUs with the authors' own library and models (Part D, D1).

DMTS (Cattin et al., JPCL 2026) runs a distilled FeNNix student every inner step and corrects it with
FeNNix-Bio1(M) every outer step. Their Algorithm 1 is BAOAB-RESPA: Langevin BAOAB inner steps with
the student, and the correction F_T - F_S as half kicks at both outer boundaries. The public FeNNol
release has their hydrogen mass repartitioning (the ``hmr`` keyword) but not the integrator, so this
script drives FeNNol's public model API (``FENNIX.load``, its preprocessing chain and the pure
``_energy_and_forces``) through Algorithm 1, the same per-step structure as FeNNol's own engine
(``fennol.md``): jitted force calls, a skin neighbour list rebuilt every ``nblist_stride`` steps.
Runs in ``.venvs/frost-fennol`` only (FeNNol is LGPL-3.0, FeNNix-Bio1 is ASL); nothing here is
imported by ``src/frost``.

Stages
- ``check``: the loop's forces against ``FENNIX.energy_and_forces`` on the box, and per-call costs;
- ``label``: FeNNix-Bio1(M) energies and forces on frames of the STS trajectory (FeNNol's engine), as
  a FeNNol training pickle;
- ``md``: STS (``--k 1``, no correction) or DMTS (``--student``, ``--k``, ``--hmr``) for ``--ps``;
  wall time per simulated ps, temperature, O-O RDF and the oxygen diffusion coefficient.

Units: A, fs, amu, eV (FeNNix-Bio1 is in eV).

Usage (guarded launcher; GPU; JAX: LD_LIBRARY_PATH and CUDA_HOME unset)
  scripts/safe_run.sh dmts_md env -u LD_LIBRARY_PATH -u CUDA_HOME XLA_PYTHON_CLIENT_PREALLOCATE=false \\
      .venvs/frost-fennol/bin/python experiments/e14_baselines/dmts_fennol.py --stage md \\
      --student data/baselines/dmts/student/best_model.fnx --k 5 --hmr 3 --ps 100 --tag dmts_k5
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data" / "baselines" / "dmts"
TEACHER = ROOT / "third_party" / "fennol-pmc" / "fennix-bio1M.fnx"
KB = 8.617333262e-5  # eV / K
ACC = 0.009648533212  # (eV / A / amu) -> A / fs^2
MASS = {1: 1.008, 8: 15.999}


def read_tinker(path: Path):
    """The FeNNol example box: Tinker xyz, one header line (natoms + comment), indexed rows."""
    lines = path.read_text().splitlines()
    n = int(lines[0].split()[0])
    rows = [ln.split() for ln in lines[1:1 + n]]
    symbols = [r[1] for r in rows]
    z = np.array([1 if s.startswith("H") else 8 for s in symbols], dtype=np.int32)
    x = np.array([[float(v) for v in r[2:5]] for r in rows])
    return z, x


def hmr_masses(z, added: float) -> np.ndarray:
    """FeNNol's ``hmr`` rule: add to every H, take the same total from the others by mass."""
    m = np.array([MASS[int(s)] for s in z])
    if added > 0:
        h = z == 1
        total = m.sum()
        m[h] += added
        m[~h] -= added * h.sum() * m[~h] / m[~h].sum()
        assert np.isclose(m.sum(), total)
    return m


class Forces:
    """One FeNNol model on the box: positions in, (energy eV, forces eV/A) out, skin neighbour list."""

    def __init__(self, path: Path, z, x0, cell, skin: float = 2.0, stride: int = 0, dt_fs: float = 1.0):
        import jax.numpy as jnp
        from flax.core import freeze, unfreeze

        from fennol import FENNIX

        self.jnp = jnp
        self.model = FENNIX.load(str(path))
        conf = {"species": np.asarray(z, np.int32), "coordinates": np.asarray(x0, np.float32),
                "batch_index": np.zeros(len(z), np.int32), "natoms": np.array([len(z)], np.int32),
                "total_charge": np.zeros(1, np.float32), "cells": np.asarray(cell, np.float32)[None],
                "reciprocal_cells": np.linalg.inv(cell).astype(np.float32)[None],
                "flags": {"minimum_image": None}}
        # the skin set-up of FeNNol's engine (fennol.md.initial.initialize_preprocessing)
        state = unfreeze(self.model.preproc_state)
        state["layers_state"] = [freeze({**unfreeze(st), "nblist_skin": skin}) for st in state["layers_state"]]
        state = freeze(state).copy({"check_input": True})
        state, conf = self.model.preprocessing(state, conf)
        self.state = state.copy({"check_input": False})
        self.conf = conf
        # FeNNol's default stride: a 2 A skin gives 40 fs at 300 K (fennol.md.integrate)
        self.stride = stride or max(1, int(skin / 2.0 * 40.0 / dt_fs))
        self.countdown = 0
        self.rebuilds = 0
        self.ef = self.model._energy_and_forces

    def __call__(self, x, steps: int = 1):
        """``steps``: MD steps since this model's previous call. The rebuild interval is a time (FeNNol's stride
        counts steps because its engine calls the model every step); a model called once per outer interval of k
        steps must still rebuild every ``stride`` steps, or atoms leave the skin between rebuilds."""
        conf = {**self.conf, "coordinates": x}
        if self.countdown <= 0:
            conf = self.model.preprocessing.process(self.state, conf)
            self.state, _, conf, _ = self.model.preprocessing.check_reallocate(self.state, conf)
            self.countdown = self.stride - steps
            self.rebuilds += 1
        else:
            conf = self.model.preprocessing.update_skin(conf)
            self.countdown -= steps
        self.conf = conf
        e, f, _ = self.ef(self.model.variables, conf)
        return e, f


def rdf_oo(frames, cell_len: float, z, r_max: float = 8.0, nbins: int = 160):
    o = np.nonzero(z == 8)[0]
    edges = np.linspace(0.0, r_max, nbins + 1)
    hist = np.zeros(nbins)
    for x in frames:
        xo = x[o]
        d = xo[:, None, :] - xo[None, :, :]
        d -= cell_len * np.round(d / cell_len)
        r = np.sqrt((d ** 2).sum(-1))[np.triu_indices(len(o), 1)]
        hist += np.histogram(r, edges)[0]
    rho = len(o) / cell_len ** 3
    shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    g = 2.0 * hist / (len(frames) * len(o) * rho * shell)
    return 0.5 * (edges[1:] + edges[:-1]), g


def diffusion_o(traj, z, every_fs: float):
    """Einstein D of oxygen (1e-5 cm^2/s) from unwrapped positions, slope over 10-50% of the window."""
    o = np.nonzero(z == 8)[0]
    x = np.asarray(traj)[:, o]
    n = len(x)
    lags = np.arange(1, n // 2)
    msd = np.array([((x[lag:] - x[:-lag]) ** 2).sum(-1).mean() for lag in lags])
    t = lags * every_fs
    sel = (t >= 0.1 * t[-1]) & (t <= t[-1])
    slope = np.polyfit(t[sel], msd[sel], 1)[0]  # A^2 / fs
    return slope / 6.0 * 1e-16 / 1e-15 / 1e-5  # A^2/fs -> cm^2/s -> 1e-5 cm^2/s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("check", "label", "md"), required=True)
    ap.add_argument("--box", default=str(BASE / "watersmall.xyz"))
    ap.add_argument("--cell", type=float, default=18.643)
    ap.add_argument("--student", default="")
    ap.add_argument("--k", type=int, default=1, help="outer interval in inner steps (1 = STS)")
    ap.add_argument("--adaptive", action="store_true",
                    help="Frost's interval controller (frost.controller.IntervalController, magnitude clause in "
                         "acceleration units) instead of a fixed k; --k is the starting interval")
    ap.add_argument("--k-max", type=int, default=16)
    ap.add_argument("--epsilon", type=float, default=2.0)
    ap.add_argument("--dt", type=float, default=1.0, help="inner step, fs")
    ap.add_argument("--hmr", type=float, default=0.0)
    ap.add_argument("--friction", type=float, default=0.001, help="per fs (DMTS: 1/ps)")
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--ps", type=float, default=20.0)
    ap.add_argument("--warm-ps", type=float, default=2.0, help="untimed, unsampled start")
    ap.add_argument("--sample-fs", type=float, default=100.0)
    ap.add_argument("--debug-outer", type=int, default=0, help="md: print T and the residual for the first N outer steps")
    ap.add_argument("--debug-every", type=int, default=0, help="md: and every N outer steps after")
    ap.add_argument("--traj", default=str(BASE / "sts" / "watersmall.traj.xyz"))
    ap.add_argument("--dump", default="", help="md: write the sampled frames to this extended-xyz file")
    ap.add_argument("--never-lose", action="store_true", help="md --adaptive: the teacher alone once k falls to 1")
    ap.add_argument("--frames", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp

    z, x0 = read_tinker(Path(args.box))
    cell = np.eye(3) * args.cell
    out_dir = BASE / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "check":
        from fennol import FENNIX

        rows = {}
        for name, path in (("teacher", TEACHER), ("student", args.student)):
            if not path:
                continue
            fn = Forces(Path(path), z, x0, cell, dt_fs=args.dt)
            e, f = fn(jnp.asarray(x0, jnp.float32))
            ref = FENNIX.load(str(path))
            e_ref, f_ref, _ = ref.energy_and_forces(species=z, coordinates=x0.astype(np.float32),
                                                     cells=cell[None].astype(np.float32),
                                                     total_charge=np.zeros(1, np.float32),
                                                     natoms=np.array([len(z)], np.int32),
                                                     flags={"minimum_image": None}, gpu_preprocessing=True)
            jax.block_until_ready(f)
            t0 = time.perf_counter()
            for _ in range(50):
                e, f = fn(jnp.asarray(x0, jnp.float32))
            jax.block_until_ready(f)
            ms = 1e3 * (time.perf_counter() - t0) / 50
            rows[name] = {"ms": ms, "max_dF_eV_A": float(np.abs(np.asarray(f) - np.asarray(f_ref)).max()),
                          "dE_eV": float(np.asarray(e).ravel()[0]) - float(np.asarray(e_ref).ravel()[0])}
            print(name, rows[name], flush=True)
        (out_dir / f"check_{args.tag}.json").write_text(json.dumps(rows, indent=1))
        return 0

    if args.stage == "label":
        from ase.io import iread

        frames = list(iread(args.traj, index=":"))
        idx = np.linspace(0, len(frames) - 1, min(args.frames, len(frames))).astype(int)
        fn = Forces(TEACHER, z, x0, cell, dt_fs=args.dt)
        fn.stride = 1  # every labelled frame gets a fresh list
        data = []
        for i in idx:
            x = frames[i].get_positions()
            e, f = fn(jnp.asarray(x, jnp.float32))
            # FeNNol's trainer reads a periodic cell from "cell" (with pbc_training: true)
            data.append({"species": z.copy(), "coordinates": x.astype(np.float32), "cell": cell.astype(np.float32),
                         "energy": float(np.asarray(e).ravel()[0]), "forces": np.asarray(f, np.float32)})
        rng = np.random.default_rng(0)
        order = rng.permutation(len(data))
        n_val = max(20, len(data) // 20)
        ds = {"training": [data[i] for i in order[n_val:]], "validation": [data[i] for i in order[:n_val]],
              "description": "FeNNix-Bio1(M) labels on FeNNol STS frames of the 216-molecule box (eV, A)"}
        path = BASE / f"labels_{args.tag}.pkl"
        with open(path, "wb") as fh:
            pickle.dump(ds, fh)
        print(f"{len(data)} frames -> {path}", flush=True)
        return 0

    # --- md: STS or DMTS (Algorithm 1) ---
    m = hmr_masses(z, args.hmr)
    minv = jnp.asarray(ACC / m, jnp.float32)[:, None]
    dt = args.dt
    c1 = float(np.exp(-args.friction * dt))
    c2 = float(np.sqrt(1.0 - c1 * c1))
    sigma = jnp.asarray(np.sqrt(KB * args.temperature * ACC / m), jnp.float32)[:, None]
    teacher = Forces(TEACHER, z, x0, cell, dt_fs=dt)
    student = Forces(Path(args.student), z, x0, cell, dt_fs=dt) if args.student else None
    fast = student or teacher
    k = args.k if student else 1
    controller = None
    if args.adaptive and student:
        import sys

        sys.path.insert(0, str(ROOT / "src"))
        from frost.controller import IntervalController  # MIT, framework-free (no torch at import)

        controller = IntervalController(epsilon=args.epsilon, k=k, k_max=args.k_max, dt_fs=dt)
    m_np = m[:, None]
    k_trace, stale_trace = [], []
    multirate = bool(student)

    @jax.jit
    def b(v, f, h):
        return v + h * f * minv

    @jax.jit
    def boa(x, v, f, key, i):
        # B (half kick from the last fast force) then A O A, the noise keyed by the step on the device: one dispatch
        v = v + (0.5 * dt) * f * minv
        x = x + 0.5 * dt * v
        v = c1 * v + c2 * sigma * jax.random.normal(jax.random.fold_in(key, i), v.shape, v.dtype)
        return x + 0.5 * dt * v, v

    rng = np.random.default_rng(args.seed)
    v = jnp.asarray(rng.normal(size=x0.shape) * np.sqrt(KB * args.temperature * ACC / m)[:, None], jnp.float32)
    x = jnp.asarray(x0, jnp.float32)
    key = jax.random.PRNGKey(args.seed)
    _, f_fast = fast(x)
    corr = jnp.zeros_like(x)
    if student:
        _, f_t = teacher(x)
        corr = f_t - f_fast
    n_warm = int(round(args.warm_ps * 1000 / dt))
    n_total = n_warm + int(round(args.ps * 1000 / dt))
    every = max(1, int(round(args.sample_fs / dt)))
    samples, temps, teacher_calls = [], [], 0
    t0 = None
    step = 0
    while step < n_total:
        if step >= n_warm and t0 is None:
            jax.block_until_ready(x)
            t0 = time.perf_counter()
        if multirate:
            v = b(v, corr, 0.5 * k * dt)  # outer half kick
        for _ in range(k):
            x, v = boa(x, v, f_fast, key, np.int32(step))
            _, f_fast = fast(x)
            v = b(v, f_fast, 0.5 * dt)
            step += 1
            if step > n_warm and step % every == 0:
                samples.append(np.asarray(x))
                temps.append(float((m[:, None] * np.asarray(v) ** 2).sum() / (3 * len(z) * KB * ACC)))
        k_used = k
        if multirate:
            _, f_t = teacher(x, steps=k_used)
            teacher_calls += 1
            new_corr = f_t - f_fast
            if controller is not None:
                # the realised staleness of the closing interval: RMS over atoms of |dR / m|, meV/(A amu)
                stale = float(np.sqrt(((np.asarray(new_corr - corr) / m_np) ** 2).sum(-1).mean()) * 1e3)
                stale_trace.append(stale)
                k = controller.observe(stale, k)
            corr = new_corr
        v = b(v, corr, 0.5 * k_used * dt)
        k_trace.append(k_used)
        if multirate and controller is not None and args.never_lose and k <= 1:
            # never lose: no multirate interval certifies, so run the teacher itself from here (latched)
            multirate, fast, f_fast, corr, k = False, teacher, f_t, jnp.zeros_like(x), 1
            print(f"never lose: teacher only from step {step}", flush=True)
        if len(k_trace) <= args.debug_outer or (args.debug_every and len(k_trace) % args.debug_every == 0):
            vv, ff, cc = np.asarray(v), np.asarray(f_fast), np.asarray(corr)
            print(f"outer {len(k_trace)} step {step}: T {float((m[:, None] * vv ** 2).sum() / (3 * len(z) * KB * ACC)):.1f} K "
                  f"| rms F_fast {np.sqrt((ff ** 2).sum(-1).mean()):.3f} | rms F_T - F_S {np.sqrt((cc ** 2).sum(-1).mean()):.3f} "
                  f"max {np.abs(cc).max():.3f} eV/A | max F_fast {np.abs(ff).max():.2f}", flush=True)
        if not np.isfinite(temps[-1] if temps else 0.0):
            print(f"non-finite at step {step}", flush=True)
            break
    jax.block_until_ready(x)
    wall = time.perf_counter() - t0
    if args.dump:
        # the sampled frames, wrapped into the cell, for the label stage (a larger and hotter distillation set)
        from ase import Atoms
        from ase.io import write

        frames_out = []
        for xs, temp in zip(samples, temps):
            if not (np.isfinite(xs).all() and temp < 1.5 * args.temperature):
                continue  # keep the student's own configurations up to a failure, not the blown-up ones
            a = Atoms(numbers=z, positions=xs, cell=cell, pbc=True)
            a.wrap()
            frames_out.append(a)
        write(args.dump, frames_out)
        print(f"{len(frames_out)} frames -> {args.dump}", flush=True)
    ps = (step - n_warm) * dt / 1000.0
    r, g = rdf_oo(samples, args.cell, z)
    d = diffusion_o(samples, z, every * dt)
    res = {"tag": args.tag, "k": k, "dt_fs": dt, "outer_fs": k * dt, "hmr": args.hmr, "friction_per_fs": args.friction,
           "student": args.student, "ps": ps, "wall_s": wall, "ms_per_fs": 1e3 * wall / (ps * 1000.0),
           "ns_per_day": ps / 1000.0 / (wall / 86400.0), "T_mean": float(np.mean(temps)), "T_std": float(np.std(temps)),
           "D_1e-5_cm2_s": d, "rdf_r": r.tolist(), "rdf_oo": g.tolist(), "teacher_calls": teacher_calls,
           "rebuilds": {"teacher": teacher.rebuilds, "student": student.rebuilds if student else None},
           "adaptive": bool(controller), "mean_k": float(np.mean(k_trace)) if k_trace else 1.0,
           "staleness_mean": float(np.mean(stale_trace)) if stale_trace else None,
           "staleness_max": float(np.max(stale_trace)) if stale_trace else None,
           "gpu": str(jax.devices()[0])}
    (out_dir / f"md_{args.tag}.json").write_text(json.dumps(res, indent=1))
    print(f"{args.tag}: k {k} dt {dt} hmr {args.hmr} | {res['ms_per_fs']:.3f} ms/fs = {res['ns_per_day']:.2f} ns/day | "
          f"T {res['T_mean']:.1f} +- {res['T_std']:.1f} K | D_O {d:.2f} e-5 cm2/s | g_OO peak {g.max():.2f} at "
          f"{r[g.argmax()]:.2f} A", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
