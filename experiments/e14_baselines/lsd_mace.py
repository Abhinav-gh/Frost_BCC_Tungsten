"""LSD (Kosmala et al., ICML 2026) with the teacher Frost uses, on our 2 x A6000 (Part D, D4).

Their copper demo (``third_party/lsd/demo.py``) pairs ORB models. This drives the same public API
(``lsd.integrators.SpeculativeLangevin`` with a ``lsd.verifiers.Verifier``) with MACE-MP-0 medium as the target
(cuEquivariance, as Frost's teacher) and MACE-MP-0 small as the draft, both as ASE calculators, so LSD and Frost
accelerate the same model on the same GPUs: the draft on GPU 0, the target's verifier workers on GPU 1, as their
demo arranges. ``--plain`` runs their non-speculative ``LangevinABOBA`` with the target alone (their baseline).
Conditions default to their demo's: 500 Cu atoms from fcc, 1500 K, 1 fs, friction 0.001/fs, 5000 warm-up and
5000 timed steps.

Output: data/results/e14/lsd_mace_<tag>.json

Usage (guarded launcher; both GPUs; .venvs/frost-lsd)
  scripts/safe_run.sh lsd_mace env -u LD_LIBRARY_PATH -u CUDA_HOME TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \\
      .venvs/frost-lsd/bin/python experiments/e14_baselines/lsd_mace.py --tag spec
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "data" / "results" / "e14"


def calculator(model: str):
    from mace.calculators import mace_mp

    return mace_mp(model=model, device="cuda", default_dtype="float32", enable_cueq=True)


def target_factory():
    return calculator("medium")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plain", action="store_true", help="their LangevinABOBA with the target alone")
    ap.add_argument("--draft", default="small", help="mace_mp model for the draft")
    ap.add_argument("--size", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1500.0)
    ap.add_argument("--friction", type=float, default=0.001, help="per fs")
    ap.add_argument("--warmup", type=int, default=5000)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--workers-per-gpu", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    from ase import units
    from ase.lattice.cubic import FaceCenteredCubic

    from lsd.ray_utils import set_driver_env_defaults

    set_driver_env_defaults()
    atoms = FaceCenteredCubic(directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], symbol="Cu",
                              size=(args.size,) * 3, pbc=True)
    rng = np.random.default_rng(args.seed)
    verifier = None
    if args.plain:
        from lsd.integrators import LangevinABOBA

        atoms.calc = target_factory()
        dyn = LangevinABOBA(atoms=atoms, timestep=units.fs, temperature_K=args.temperature,
                            friction=args.friction / units.fs, rng=rng)
    else:
        import ray

        from lsd.integrators import SpeculativeLangevin
        from lsd.ray_utils import init_head_node
        from lsd.verifiers import Verifier

        init_head_node(draft_needs_gpu=True)
        print(f"Ray cluster resources: {ray.cluster_resources()}", flush=True)
        atoms.calc = calculator(args.draft)
        verifier = Verifier(target_model_factory=target_factory, num_workers_per_gpu=args.workers_per_gpu)
        dyn = SpeculativeLangevin(atoms=atoms, verifier=verifier, timestep=units.fs,
                                  temperature_K=args.temperature, friction=args.friction / units.fs)
    temps = []
    dyn.attach(lambda: temps.append(atoms.get_kinetic_energy() / (1.5 * units.kB * len(atoms))), interval=100)
    dyn.run(steps=args.warmup)
    if verifier is not None:
        verifier.reset_statistics()
        verifier.reset_step_buffers()
    t0 = time.time()
    dyn.run(steps=args.steps)
    wall = time.time() - t0
    out = {"tag": args.tag, "mode": "plain" if args.plain else "speculative", "target": "mace-mp-0 medium (cueq)",
           "draft": None if args.plain else f"mace-mp-0 {args.draft}", "n_atoms": len(atoms),
           "temperature_K": args.temperature, "friction_per_fs": args.friction, "steps": args.steps,
           "wall_s": wall, "ms_per_fs": 1e3 * wall / args.steps, "T_mean_timed": float(np.mean(temps[-args.steps // 100:])),
           "verifier": (verifier.get_statistics() if verifier is not None else None)}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"lsd_mace_{args.tag}.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
