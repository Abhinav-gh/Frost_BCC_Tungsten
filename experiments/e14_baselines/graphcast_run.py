"""GraphCast rerun on our GPU (Lam et al., Science 2023): their released GraphCast_small (1 degree, 13 levels), their
normalisation statistics and their ERA5 2022-01-01 example, run exactly as their demo notebook composes it (now packaged
as ``weathernext``): GraphCast -> Bfloat16Cast -> InputsAndResiduals -> autoregressive Predictor, rolled out with
``rollout.chunked_prediction``.

Reports the wall time of a 10-day forecast (40 six-hour steps) after a compile run, and the RMSE of 500 hPa geopotential
and 2 m temperature against the ERA5 targets at day 1, 5 and 10. GraphCast is a whole-state emulator with no physics /
dynamics split: a weather cost/skill reference, not a model Frost accelerates.

Environment: .venvs/frost-graphcast (JAX; run with LD_LIBRARY_PATH and CUDA_HOME unset). Code Apache-2.0; the
weights are CC BY-NC-SA 4.0 and stay in data/ (never redistributed).

Usage
  env -u LD_LIBRARY_PATH -u CUDA_HOME .venvs/frost-graphcast/bin/python experiments/e14_baselines/graphcast_run.py
"""

from __future__ import annotations

import dataclasses
import functools
import json
import time
from pathlib import Path

import haiku as hk
import jax
import numpy as np
import xarray
from weathernext.utils import autoregressive, casting, checkpoint, data_utils, normalization, rollout
from weathernext.weathernext1_graph import graphcast

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data" / "weather" / "graphcast"
OUT = ROOT / "data" / "results" / "e14"


def main() -> int:
    with open(D / "params" / "GraphCast_small.npz", "rb") as fh:
        ckpt = checkpoint.load(fh, graphcast.CheckPoint)
    params, state = ckpt.params, {}
    model_config, task_config = ckpt.model_config, ckpt.task_config
    batch = xarray.load_dataset(D / "dataset" / "era5_2022-01-01_res-1.0_levels-13_steps-40.nc").compute()
    task = {k: v for k, v in dataclasses.asdict(task_config).items()
            if k in ("input_variables", "target_variables", "forcing_variables", "pressure_levels", "input_duration")}
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        batch, target_lead_times=slice("6h", f"{40 * 6}h"), **task)
    stats = {n: xarray.load_dataset(D / "stats" / f"{n}.nc").compute()
             for n in ("diffs_stddev_by_level", "mean_by_level", "stddev_by_level")}

    def construct(model_config, task_config):
        predictor = graphcast.GraphCast(model_config, task_config)
        predictor = casting.Bfloat16Cast(predictor)
        predictor = normalization.InputsAndResiduals(predictor, **stats)
        return autoregressive.Predictor(predictor, gradient_checkpointing=True)

    @hk.transform_with_state
    def run_forward(model_config, task_config, inputs, targets_template, forcings):
        return construct(model_config, task_config)(inputs, targets_template=targets_template, forcings=forcings)

    apply = jax.jit(functools.partial(run_forward.apply, model_config=model_config, task_config=task_config))

    def predictor_fn(**kw):
        return apply(params=params, state=state, **kw)[0]

    def forecast():
        return rollout.chunked_prediction(predictor_fn, rng=jax.random.PRNGKey(0), inputs=inputs,
                                          targets_template=targets * np.nan, forcings=forcings)

    t0 = time.perf_counter()
    forecast()  # compile
    compile_s = time.perf_counter() - t0
    walls = []
    for _ in range(2):
        t0 = time.perf_counter()
        pred = forecast()
        jax.block_until_ready(jax.tree_util.tree_leaves(pred.to_array().data))
        walls.append(time.perf_counter() - t0)
    report = {"model": ckpt.description, "license": ckpt.license, "device": str(jax.devices()[0]),
              "steps": 40, "compile_plus_first_s": compile_s, "forecast_10d_s": walls,
              "ms_per_6h_step": 1e3 * min(walls) / 40, "rmse": {}}
    weights = np.cos(np.deg2rad(targets.lat))
    for var, sel in (("geopotential", {"level": 500}), ("2m_temperature", {})):
        p, t = pred[var].sel(**sel), targets[var].sel(**sel)
        err = ((p - t) ** 2).weighted(weights).mean(("lat", "lon"))
        rmse = np.sqrt(err.squeeze().values)
        report["rmse"][var + ("_500" if sel else "")] = {f"day{d}": float(rmse[4 * d - 1]) for d in (1, 5, 10)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "graphcast_small_2022-01-01.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
