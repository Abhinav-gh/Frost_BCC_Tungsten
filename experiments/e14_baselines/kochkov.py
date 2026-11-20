"""Kochkov et al. (PNAS 2021) rerun on our GPU with their code and their trained models (Part D, D5).

Their three released models for 2D Kolmogorov flow at Re 1000 on a 64x64 grid (learned interpolation LI,
learned correction LC, encode-process-decode EPD; ``gs://gresearch/jax-cfd/public_models``) are rolled
out with jax-cfd's own model builder and gin configs, exactly as their demo notebook does, from the
initial conditions of their public evaluation set, and scored with jax-cfd's own
``compute_summary_dataset`` against their 2048x2048 reference (downsampled to 64x64). Their own
finite-volume baselines at 64-1024 (the same file set) are scored the same way. Reported per model:
the sample-mean vorticity correlation over time, the time it stays above 0.95 (their headline metric),
and the cost of one model step at a batch of ``--samples`` trajectories on one A6000.

Runs in ``.venvs/frost-jaxcfd`` (jax 0.4.30, dm-haiku 0.0.12, jax-cfd from its repository; Apache-2.0).

Output: data/results/e14/kochkov_<tag>.json

Usage (guarded launcher; GPU)
  scripts/safe_run.sh kochkov env -u LD_LIBRARY_PATH -u CUDA_HOME XLA_PYTHON_CLIENT_PREALLOCATE=false \\
      .venvs/frost-jaxcfd/bin/python experiments/e14_baselines/kochkov.py --tag re1000
"""

from __future__ import annotations

import argparse
import functools
import json
import pickle
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data" / "baselines" / "kochkov"
RES = ROOT / "data" / "results" / "e14"


class CheckpointState:
    """The class their checkpoints were pickled with (jax-cfd's demo notebook defines it the same way)."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "CheckpointState":
            return CheckpointState
        return super().find_class(module, name)


def strip_imports(s: str) -> str:
    return "\n".join(line for line in s.splitlines() if not line.startswith("import"))


def tau_095(corr: np.ndarray, times: np.ndarray) -> float:
    below = np.nonzero(corr < 0.95)[0]
    return float(times[below[0]]) if len(below) else float(times[-1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=BASE, help="their eval files (re4000/ for Re 4000)")
    ap.add_argument("--grid", type=int, default=64, help="the eval files' output grid (Re 4000: 128)")
    ap.add_argument("--extra", default="",
                    help="name:file of their own released rollouts scored as-is (Re 4000: "
                         "LI:learned_interpolation_long_eval_128x128_128x128.nc)")
    ap.add_argument("--models", default="LI,LC,EPD")
    ap.add_argument("--baselines", default="64,128,256,512,1024")
    ap.add_argument("--reference", type=int, default=2048)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--length", type=int, default=200, help="saved frames (each 10 model steps)")
    ap.add_argument("--inner-steps", type=int, default=10)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    import gin
    import haiku as hk
    import jax
    import xarray

    import jax_cfd.data as cfd_data
    from jax_cfd.data import evaluation
    from jax_cfd.ml import model_builder, physics_specifications

    def open_ds(n):
        return xarray.open_dataset(args.data_dir / f"eval_{n}x{n}_{args.grid}x{args.grid}.nc")

    ref_full = open_ds(args.reference)
    n_samples = min(args.samples, ref_full.sizes["sample"])
    sel = {"sample": slice(0, n_samples), "time": slice(0, args.length)}
    reference = ref_full.isel(**sel).load()
    grid = cfd_data.xarray_utils.grid_from_attrs(ref_full.attrs)
    ics = tuple(reference[name].isel(time=slice(0, 1)).values
                for name in cfd_data.xarray_utils.XR_VELOCITY_NAMES[:grid.ndim])
    times = reference["time"].values - reference["time"].values[0]
    out = {"dataset": str(args.data_dir.name), "grid": args.grid, "samples": n_samples, "length": args.length,
           "frame_dt": float(times[1]), "gpu": str(jax.devices()[0]), "models": {}, "baselines": {}}

    def summarise(ds):
        summary = evaluation.compute_summary_dataset(ds, reference)
        corr = np.asarray(summary.vorticity_correlation.values)
        return corr, tau_095(corr, times)

    for n in [int(b) for b in args.baselines.split(",") if b]:
        ds = open_ds(n).isel(**sel).load()
        ds = ds.assign_coords(x=reference.x, y=reference.y, time=reference.time)
        corr, tau = summarise(ds)
        out["baselines"][f"fv_{n}"] = {"tau": tau, "corr": corr.tolist()}
        print(f"baseline FV {n}x{n}: tau {tau:.2f}", flush=True)

    for item in (e for e in args.extra.split(",") if e):
        name, fname = item.split(":")
        ds = xarray.open_dataset(args.data_dir / fname).isel(**sel).load()
        ds = ds.assign_coords(x=reference.x, y=reference.y, time=reference.time)
        corr, tau = summarise(ds)
        out["baselines"][name] = {"tau": tau, "corr": corr.tolist(), "source": fname}
        print(f"released rollout {name}: tau {tau:.2f}", flush=True)

    for name in (m for m in args.models.split(",") if m):
        with open(BASE / f"{name}_ckpt.pkl", "rb") as fh:
            ckpt = _Unpickler(fh).load()
        gin.clear_config()
        gin.parse_config(ckpt.model_config_str)
        gin.parse_config(strip_imports(ref_full.attrs["physics_config_str"]))
        dt = float(ckpt.model_time_step)
        physics_specs = physics_specifications.get_physics_specs()
        model_cls = model_builder.get_model_cls(grid, dt, physics_specs)

        def compute_trajectory_fwd(x, model_cls=model_cls):
            solver = model_cls()
            x = solver.encode(x)
            _, trajectory = solver.trajectory(x, args.length, args.inner_steps, start_with_input=True,
                                              post_process_fn=solver.decode)
            return trajectory

        model = hk.without_apply_rng(hk.transform(compute_trajectory_fwd))
        fn = jax.jit(jax.vmap(functools.partial(model.apply, ckpt.eval_params)))
        pred = jax.block_until_ready(fn(ics))  # compile and run
        t0 = time.perf_counter()
        pred = jax.block_until_ready(fn(ics))
        wall = time.perf_counter() - t0
        steps = (args.length - 1) * args.inner_steps
        ds = cfd_data.xarray_utils.velocity_trajectory_to_xarray(pred, grid, samples=True)
        ds = ds.assign_coords(x=reference.x, y=reference.y, time=reference.time)
        corr, tau = summarise(ds)
        ms_step = 1e3 * wall / steps
        out["models"][name] = {"tau": tau, "corr": corr.tolist(), "model_dt": dt, "steps": steps,
                               "ms_per_step_batch": ms_step, "ms_per_step_per_traj": ms_step / n_samples,
                               "ms_per_time_unit_per_traj": ms_step / n_samples / dt}
        print(f"{name}: tau {tau:.2f} | {ms_step:.3f} ms per step at batch {n_samples} "
              f"({ms_step / n_samples / dt:.2f} ms per time unit per trajectory)", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"kochkov_{args.tag}.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
