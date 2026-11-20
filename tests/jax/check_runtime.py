"""CPU checks of the runtime's JAX backend (frost.runtime.hold) on a 16^3 grid.

1. The executor driving ``HoldBackend`` in sync mode reproduces ``split3d.advance`` (the shipped
   hold loop) to round-off, for a closure with a non-zero correction.
2. Modes "extrap" and "spec" run, stay finite, and differ from sync (they change the held value).
3. A hold-error clause that rejects every interval longer than one step rolls the run back to
   k = 1, which equals the every-step closure (k = 1 in ``split3d.advance``).

Usage
  env -u LD_LIBRARY_PATH -u CUDA_HOME JAX_PLATFORMS=cpu PYTHONPATH=src \\
      .venvs/frost-mhd/bin/python tests/jax/check_runtime.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from frost.cfd import spectral3d as s3, split3d
from frost.cfd.spectral3d import Grid3
from frost.runtime import Executor, HoldError
from frost.runtime.hold import HoldBackend


def main() -> int:
    g = Grid3(16, nu=3e-3, power=0.1, dealias="exp")
    model = split3d.init_model(jax.random.PRNGKey(1), hidden=8, blocks=1, scale_out=0.05)
    # a non-zero decoder, so the correction (and the hold) matters
    dec = model["params"]["dec"]
    model["params"]["dec"] = {"w": 0.05 * jax.random.normal(jax.random.PRNGKey(2), dec["w"].shape), "b": dec["b"]}
    u0 = s3.random_initial(g, jax.random.PRNGKey(0), batch=(2,))
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        fails += 0 if ok else 1
        print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}", flush=True)

    ref = split3d.advance(g, model, u0, k=4, dt=0.04, n_steps=16)
    b = HoldBackend(g, model, u0, dt=0.04)
    Executor(b, [], k=4).run(16)
    err = float(jnp.max(jnp.abs(b.u - ref)) / jnp.max(jnp.abs(ref)))
    check("sync executor == split3d.advance (k 4, 16 steps)", err < 1e-5, f"rel {err:.2e}")

    outs = {}
    for mode in ("extrap", "spec"):
        bm = HoldBackend(g, model, u0, dt=0.04)
        Executor(bm, [], k=4, mode=mode).run(16)
        outs[mode] = bm.u
        finite = bool(jnp.all(jnp.isfinite(bm.u.real)))
        diff = float(jnp.max(jnp.abs(bm.u - ref)) / jnp.max(jnp.abs(ref)))
        check(f"mode {mode} finite and different from sync", finite and diff > 1e-7, f"rel diff {diff:.2e}")

    every = split3d.advance(g, model, u0, k=1, dt=0.04, n_steps=8)
    br = HoldBackend(g, model, u0, dt=0.04)
    ex = Executor(br, [HoldError(e_star=0.0)], k=8)
    ex.run(8)
    ks = [r[1] for r in ex.log]
    err1 = float(jnp.max(jnp.abs(br.u - every)) / jnp.max(jnp.abs(every)))
    check("hold-error clause rolls back to k = 1 (== every step)", ks == [1] * 8 and err1 < 1e-5,
          f"ks {ks} rel {err1:.2e}")
    print(f"{'all ok' if not fails else f'{fails} FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    raise SystemExit(main())
