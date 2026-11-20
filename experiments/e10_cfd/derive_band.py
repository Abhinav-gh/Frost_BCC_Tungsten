"""A coarser band of a stored DNS, cut spectrally from a finer stored band.

``hit_dns.py`` stores the DNS truncated to each coarse grid it was given (``train/u<n>``,
each under that grid's exponential filter). A grid not stored can be cut exactly from any
finer stored band: truncating the 48^3 band to 40^3 keeps |k| <= 19, where the 48^3 filter
is 1 to 3e-3, and applies the 40^3 filter, which is what the DNS truncated directly to
40^3 would hold. Written to a separate file (the DNS file is never modified), with the
source's attributes, for ``w3split.py --train-data``.

Usage (guarded launcher; check lane, CPU, seconds)
  LANE=check scripts/safe_run.sh derive_u40 env -u LD_LIBRARY_PATH -u CUDA_HOME JAX_PLATFORMS=cpu \\
      PYTHONPATH=src .venvs/frost-mhd/bin/python experiments/e10_cfd/derive_band.py \\
      --src data/cfd/hit288_nu1p5.h5 --from 48 --to 40 --out data/cfd/hit288_nu1p5_u40.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np

from frost.cfd import spectral3d as s3
from frost.cfd.spectral3d import Grid3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--from", dest="src_n", type=int, required=True)
    ap.add_argument("--to", dest="dst_n", type=int, required=True)
    ap.add_argument("--group", default="train")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunk", type=int, default=25)
    args = ap.parse_args()
    if args.dst_n >= args.src_n:
        raise SystemExit("--to must be coarser than --from")
    if args.out.exists():
        raise SystemExit(f"{args.out} exists")

    with h5py.File(args.src, "r") as src, h5py.File(args.out, "w") as dst:
        attrs = dict(src.attrs)
        dst.attrs.update({**attrs, "derived_from": f"{args.src.name}:{args.group}/u{args.src_n}"})
        fine = Grid3(args.src_n, nu=float(attrs["nu"]), power=float(attrs["power"]), dealias="exp")
        coarse = Grid3(args.dst_n, nu=float(attrs["nu"]), power=float(attrs["power"]), dealias="exp")
        u = src[f"{args.group}/u{args.src_n}"]
        n_t = u.shape[0]
        out = dst.create_dataset(f"{args.group}/u{args.dst_n}", (n_t, *u.shape[1:3], *(args.dst_n,) * 3), "f4")
        for t0 in range(0, n_t, args.chunk):
            block = jnp.asarray(u[t0:t0 + args.chunk])
            cut = s3.to_real(coarse, s3.truncate(fine, coarse, s3.to_hat(fine, block)))
            out[t0:t0 + block.shape[0]] = np.asarray(cut)
        if f"{args.group}/stats" in src:
            dst.create_dataset(f"{args.group}/stats", data=np.asarray(src[f"{args.group}/stats"]))
        print(f"wrote {args.out}: {args.group}/u{args.dst_n} {out.shape} from u{args.src_n}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
