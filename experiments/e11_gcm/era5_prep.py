"""G1 data: ERA5 initial conditions and verification fields on the NeuralGCM grids.

From the public ARCO-ERA5 store (0.25 degree, hourly, 37 levels; anonymous access), as in
NeuralGCM's inference demo, for January 2020 (held out of NeuralGCM's training):
- **initial conditions** at the ``--inits`` times: every input variable on all 37 levels,
  plus the forcings (SST, sea ice) shifted 24 hours back as the demo does;
- **forcings** daily over the window (persistence or daily SST for the forecast);
- **verification** every 6 hours over the window: z500, t850, q700, u850, v850 (the
  WeatherBench2 headline variables, named ``<variable>_<level>``), on the same grids.

Each is regridded conservatively (dinosaur's ``ConservativeRegridder``) onto the data grid
of every checkpoint given, NaNs filled by nearest neighbour, and saved as NetCDF.

Output: data/gcm/era5_<grid>_<tag>_{init,verif}.nc

Usage (guarded launcher; network, frost-gcm env)
  LANE=download scripts/safe_run.sh gcm_era5 env -u LD_LIBRARY_PATH -u CUDA_HOME JAX_PLATFORMS=cpu \\
      PYTHONPATH=src .venvs/frost-gcm/bin/python experiments/e11_gcm/era5_prep.py --tag jan2020
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import neuralgcm
import numpy as np
import xarray
from dinosaur import horizontal_interpolation, spherical_harmonic, xarray_utils

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "gcm"
ERA5 = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
VERIF = {"geopotential": 500, "temperature": 850, "specific_humidity": 700,
         "u_component_of_wind": 850, "v_component_of_wind": 850}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="deterministic_2_8_deg,deterministic_1_4_deg")
    ap.add_argument("--inits", default="2020-01-01T00,2020-01-04T00,2020-01-07T00,2020-01-10T00")
    ap.add_argument("--start", default="2020-01-01T00")
    ap.add_argument("--end", default="2020-01-21T00")
    ap.add_argument("--tag", default="jan2020")
    ap.add_argument("--verif", default=",".join(f"{k}:{v}" for k, v in VERIF.items()),
                    help="verification fields as variable:level, comma-separated")
    ap.add_argument("--init-chunk", type=int, default=8, help="starts fetched per request (RAM: ~1.1 GB each)")
    ap.add_argument("--parts", default="init,forcing,verif",
                    help="which files to write (training starts need only init)")
    args = ap.parse_args()
    parts = args.parts.split(",")
    verif_fields = {k: int(v) for k, v in (item.split(":") for item in args.verif.split(","))}

    full = xarray.open_zarr(ERA5, chunks=None, storage_options={"token": "anon"})
    src_grid = spherical_harmonic.Grid(
        latitude_nodes=full.sizes["latitude"], longitude_nodes=full.sizes["longitude"],
        latitude_spacing=xarray_utils.infer_latitude_spacing(full.latitude),
        longitude_offset=xarray_utils.infer_longitude_offset(full.longitude))
    models = {}
    for name in args.models.split(","):
        with open(OUT / "models" / f"{name}.pkl", "rb") as fh:
            models[name] = neuralgcm.PressureLevelModel.from_checkpoint(pickle.load(fh))
    any_model = next(iter(models.values()))
    inputs, forcing_vars = any_model.input_variables, any_model.forcing_variables

    regridders = {name: horizontal_interpolation.ConservativeRegridder(src_grid, m.data_coords.horizontal, skipna=True)
                  for name, m in models.items()}

    def to_grids(ds):
        return {name: xarray_utils.fill_nan_with_nearest(xarray_utils.regrid(ds, r)) for name, r in regridders.items()}

    # initial conditions in chunks: one 0.25-degree start is ~1.1 GB, regridded ~10-40 MB
    t0 = time.perf_counter()
    inits = [np.datetime64(t) for t in args.inits.split(",")]
    pieces = {name: [] for name in models}
    for i in range(0, len(inits), args.init_chunk):
        ds = (full[inputs + forcing_vars]
              .pipe(xarray_utils.selective_temporal_shift, variables=forcing_vars, time_shift="24 hours")
              .sel(time=inits[i:i + args.init_chunk]).compute())
        for name, out in to_grids(ds).items():
            pieces[name].append(out)
        print(f"initial conditions {i + ds.sizes['time']}/{len(inits)}, {time.perf_counter() - t0:.0f} s", flush=True)
        del ds
    out_parts = {name: {"init": xarray.concat(v, "time")} for name, v in pieces.items()}

    window = slice(np.datetime64(args.start), np.datetime64(args.end))
    t0 = time.perf_counter()
    if "forcing" in parts:
        for name, out in to_grids(full[forcing_vars].sel(time=window).resample(time="1D").first().compute()).items():
            out_parts[name]["forcing"] = out
    if "verif" in parts:
        ds = xarray.Dataset({f"{k}_{lev}": full[k].sel(level=lev, time=window).thin(time=6).drop_vars("level")
                             for k, lev in verif_fields.items()}).compute()
        for name, out in to_grids(ds).items():
            out_parts[name]["verif"] = out
    print(f"forcing and verification ({args.parts}): {time.perf_counter() - t0:.0f} s", flush=True)

    for name, model in models.items():
        grid = model.data_coords.horizontal
        label = f"{grid.longitude_nodes}x{grid.latitude_nodes}"
        for part, out in out_parts[name].items():
            path = OUT / f"era5_{label}_{args.tag}_{part}.nc"
            out.to_netcdf(path)
            print(f"{name}: wrote {path.name} {dict(out.sizes)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
