"""CPU checks for frost.gcm (the frost-gcm environment has no pytest).

Run: env -u LD_LIBRARY_PATH -u CUDA_HOME JAX_PLATFORMS=cpu PYTHONPATH=src \
         .venvs/frost-gcm/bin/python tests/jax/check_gcm.py
Prints one line per check and "ok N checks" last; exits non-zero on any failure.
Small grids only, so CPU is allowed (CLAUDE.md: tiny unit-test models).
"""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
from dinosaur import spherical_harmonic

from frost.gcm.neuralgcm_split import SplitModel, fft_transforms
from frost.gcm.transforms import F32RealSphericalHarmonics, FFTSphericalHarmonics

FAILED: list[str] = []
PASSED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASSED if cond else FAILED).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'} {name} {detail}")


def fft_transforms_match_real():
    worst = {FFTSphericalHarmonics: 0.0, F32RealSphericalHarmonics: 0.0}
    for kw in ({"longitude_wavenumbers": 22, "dealiasing": "linear"},
               {"longitude_wavenumbers": 22, "dealiasing": "quadratic"},
               {"longitude_wavenumbers": 32, "dealiasing": "cubic"}):
        grids = [spherical_harmonic.Grid.with_wavenumbers(latitude_spacing="gauss", spherical_harmonics_impl=impl,
                                                          radius=1.0, **kw)
                 for impl in (spherical_harmonic.RealSphericalHarmonics, *worst)]
        real, *ours = (g.spherical_harmonics for g in grids)
        key = jax.random.key(kw["longitude_wavenumbers"])
        modal = jax.random.normal(key, (3, *real.modal_shape)) * real.mask
        nodal = jax.random.normal(jax.random.fold_in(key, 1), (3, *real.nodal_shape))
        scale = max(float(jnp.max(jnp.abs(real.inverse_transform(modal)))), float(jnp.max(jnp.abs(real.transform(nodal)))))
        for impl, sh in zip(worst, ours, strict=True):
            inv = float(jnp.max(jnp.abs(sh.inverse_transform(modal) - real.inverse_transform(modal))))
            fwd = float(jnp.max(jnp.abs(sh.transform(nodal) - real.transform(nodal))))
            worst[impl] = max(worst[impl], inv / scale, fwd / scale)
        shape = real.nodal_shape
    for impl, err in worst.items():
        check(f"{impl.__name__} matches RealSphericalHarmonics (forward, inverse; linear, quadratic, cubic grids)",
              err < 1e-5, f"(worst relative {err:.1e}; last nodal grid {shape})")


def leaves(tree):
    return [v for v in jax.tree_util.tree_leaves(tree) if jnp.issubdtype(jnp.asarray(v).dtype, jnp.inexact)]


def split_reproduces_advance():
    # NeuralGCM's bundled toy checkpoint (TL63, stochastic) and demo data: tiny, CPU is fine. It is
    # configured with FastSphericalHarmonics, so the FFT override must be refused, not applied.
    import neuralgcm

    ckpt = neuralgcm.demo.load_checkpoint_tl63_stochastic()
    m = SplitModel(ckpt)
    data = neuralgcm.demo.load_data(m.api.data_coords)
    forcings = m.api.forcings_from_xarray(data.isel(time=0))
    x0 = m.api.encode(m.api.inputs_from_xarray(data.isel(time=0)), forcings, jax.random.key(0))
    x = x0
    for _ in range(m.num_substeps):
        f = m.forcing_at(m.params, forcings, x)
        x = m.finish(m.params, x, m.physics(m.params, x, f), f)
    ref = m.api.advance(x0, forcings)
    num = sum(float(jnp.sum((a - b) ** 2)) for a, b in zip(leaves(x), leaves(ref), strict=True))
    den = sum(float(jnp.sum(b**2)) for b in leaves(ref))
    err = (num / den) ** 0.5
    try:
        SplitModel(ckpt, fft_transforms())
        refused = False
    except ValueError:
        refused = True
    check("SplitModel: physics then held-tendency dycore reproduces NeuralGCM's advance; FFT refused on a "
          "fast-layout checkpoint", err < 1e-6 and refused, f"(relative L2 {err:.1e}, refused {refused})")


if __name__ == "__main__":
    for fn in (fft_transforms_match_real, split_reproduces_advance):
        fn()
    if FAILED:
        print(f"FAILED {len(FAILED)}: {', '.join(FAILED)}")
        sys.exit(1)
    print(f"ok {len(PASSED)} checks")
