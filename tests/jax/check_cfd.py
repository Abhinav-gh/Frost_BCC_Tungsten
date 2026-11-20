"""CPU checks for frost.cfd (the frost-mhd environment has no pytest).

Run: env -u LD_LIBRARY_PATH -u CUDA_HOME JAX_PLATFORMS=cpu PYTHONPATH=src \
         .venvs/frost-mhd/bin/python tests/jax/check_cfd.py
Prints one line per check and "ok N checks" last; exits non-zero on any failure.
Tiny grids only (n = 32), so CPU is allowed (CLAUDE.md: tiny unit-test models).
"""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np

from frost.cfd import closure, finite_volume as fv, mhd3d, pipeline, pipeline3d, schedule, spectral, spectral3d as s3, split, split3d, tiles
from frost.cfd.spectral import Grid

FAILED: list[str] = []
PASSED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASSED if cond else FAILED).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'} {name} {detail}")


def single_mode_decays_exactly():
    # cos(x) cos(y) is a steady Euler solution; with no forcing or drag it decays as exp(-2 nu t)
    g = Grid(32, nu=0.01, alpha=0.0, k_force=0)
    x, y = g.coords
    w0 = jnp.asarray(np.cos(x) * np.cos(y), jnp.float32)[None]
    wh = spectral.to_hat(g, w0)
    for _ in range(50):
        wh = spectral.step(g, wh, 0.01)
    got = spectral.to_real(g, wh)
    want = w0 * np.exp(-2 * 0.01 * 0.5)
    err = float(jnp.max(jnp.abs(got - want)))
    check("single Fourier mode decays at exp(-nu k^2 t)", err < 1e-5, f"(max err {err:.1e})")


def inviscid_invariants():
    g = Grid(32, nu=0.0, alpha=0.0, k_force=0)
    wh = spectral.random_initial(g, jax.random.PRNGKey(0), batch=(1,))

    def invariants(w):
        e = float(spectral.energy_spectrum(g, w).sum())
        z = float(jnp.mean(spectral.to_real(g, w) ** 2))
        return e, z

    e0, z0 = invariants(wh)
    for _ in range(100):
        wh = spectral.step(g, wh, 0.002)
    e1, z1 = invariants(wh)
    check("Galerkin truncation conserves energy and enstrophy (inviscid)",
          abs(e1 / e0 - 1) < 1e-4 and abs(z1 / z0 - 1) < 1e-4,
          f"(dE/E {e1 / e0 - 1:.1e}, dZ/Z {z1 / z0 - 1:.1e})")


def spectrum_sums_to_energy():
    g = Grid(32)
    wh = spectral.random_initial(g, jax.random.PRNGKey(1), batch=(1,))
    u, v = spectral.velocity(g, wh)
    ke = float(jnp.mean(0.5 * (u**2 + v**2)))
    es = float(spectral.energy_spectrum(g, wh, kbins=32).sum())
    check("energy spectrum sums to the kinetic energy", abs(es / ke - 1) < 1e-4, f"({es:.5f} vs {ke:.5f})")


def truncation_is_a_filter():
    fine, coarse = Grid(64), Grid(32)
    w_c = spectral.random_initial(coarse, jax.random.PRNGKey(2), batch=(1,))
    # embed the coarse band in the fine grid by zero padding, then truncate back
    nf, nc, kc = 64, 32, coarse.kmax
    s = (nf / nc) ** 2
    w_f = jnp.zeros((1, nf, nf // 2 + 1), w_c.dtype)
    w_f = w_f.at[..., : kc + 1, : kc + 1].set(w_c[..., : kc + 1, : kc + 1] * s)
    w_f = w_f.at[..., nf - kc :, : kc + 1].set(w_c[..., nc - kc :, : kc + 1] * s)
    back = spectral.truncate(fine, coarse, w_f)
    err = float(jnp.max(jnp.abs(back - w_c)) / jnp.max(jnp.abs(w_c)))
    # and the real fields agree at the coarse points
    rf = spectral.to_real(fine, w_f)[..., ::2, ::2]
    rc = spectral.to_real(coarse, w_c)
    err2 = float(jnp.max(jnp.abs(rf - rc)) / jnp.max(jnp.abs(rc)))
    check("spectral truncation inverts band-limited zero padding", err < 1e-6 and err2 < 1e-5,
          f"({err:.1e}, {err2:.1e})")


def model_for(g: Grid, key=3):
    params = closure.init_cnn(jax.random.PRNGKey(key), hidden=8, layers=3)
    # give the last layer weights so the closure is not identically zero
    params[-1]["w"] = jax.random.normal(jax.random.PRNGKey(key + 1), params[-1]["w"].shape) * 0.05
    return {"params": params, "scale": {"u": 1.0, "w": 5.0}}


def flux_closure_conserves():
    g = Grid(32)
    model = model_for(g)
    wh = spectral.random_initial(g, jax.random.PRNGKey(4), batch=(2,))
    tend = closure.apply_operator(g, closure.coefficients(g, model, wh), wh)
    mean = float(jnp.max(jnp.abs(tend[..., 0, 0])))
    size = float(jnp.max(jnp.abs(tend)))
    check("flux-form closure has zero mean tendency", mean < 1e-6 * max(size, 1.0) and size > 0,
          f"(mean mode {mean:.1e}, max {size:.2e})")


def operator_and_value_agree_at_one_stage():
    g = Grid(32)
    model = model_for(g)
    wh = spectral.random_initial(g, jax.random.PRNGKey(5), batch=(1,))
    coef = closure.coefficients(g, model, wh)
    a = closure.apply_operator(g, coef, wh)
    b = schedule._refresh(g, model, wh, "value", 0.17)
    check("value hold = operator applied at the refresh state", bool(jnp.allclose(a, b, atol=1e-5)))


def hold_with_k1_matches_manual_loop():
    g = Grid(32)
    model = model_for(g)
    wh = spectral.random_initial(g, jax.random.PRNGKey(6), batch=(1,))
    got = schedule.advance(g, model, wh, mode="operator", k=1, dt=0.01, n_outer=5)
    ref = wh
    for _ in range(5):
        coef = closure.coefficients(g, model, ref)
        ref = spectral.step(g, ref, 0.01, lambda w, c=coef: closure.apply_operator(g, c, w))
    err = float(jnp.max(jnp.abs(got - ref)) / jnp.max(jnp.abs(ref)))
    check("operator schedule at k = 1 is the every-step ML solver", err < 1e-5, f"({err:.1e})")


def held_value_is_held():
    g = Grid(32)
    model = model_for(g)
    wh = spectral.random_initial(g, jax.random.PRNGKey(7), batch=(1,))
    got = schedule.advance(g, model, wh, mode="value", k=4, dt=0.01, n_outer=1)
    tend = schedule._refresh(g, model, wh, "value", 0.17)
    ref = wh
    for _ in range(4):
        ref = spectral.step(g, ref, 0.01, lambda w: tend)
    err = float(jnp.max(jnp.abs(got - ref)) / jnp.max(jnp.abs(ref)))
    check("value schedule at k = 4 holds the refresh tendency", err < 1e-5, f"({err:.1e})")


def advection_transports():
    g = Grid(32)
    f = jax.random.normal(jax.random.PRNGKey(8), (1, 32, 32, 2))
    zero = jnp.zeros((1, 32, 17), jnp.complex64)
    same = closure.advect(g, f, zero, 0.1)
    check("advecting by zero velocity is the identity", bool(jnp.allclose(same, f, atol=1e-6)))
    # u = const 1 in x from psi = y: omega = 0, so build the velocity through a stream function is
    # impossible with zero mean vorticity; check the interpolation directly instead
    shifted = jax.vmap(jax.vmap(lambda c: jax.scipy.ndimage.map_coordinates(
        c, list(jnp.meshgrid(jnp.arange(32.0) - 1.0, jnp.arange(32.0), indexing="ij")),
        order=1, mode="wrap"), in_axes=-1, out_axes=-1))(f)
    check("bilinear periodic shift by one cell is a roll",
          bool(jnp.allclose(shifted, jnp.roll(f, 1, axis=1), atol=1e-6)))


def smagorinsky_dissipates():
    g = Grid(32)
    wh = spectral.random_initial(g, jax.random.PRNGKey(9), batch=(1,))
    tend = closure.smagorinsky(g, wh)
    w = spectral.to_real(g, wh)
    dz = float(jnp.mean(w * spectral.to_real(g, tend)))
    check("Smagorinsky removes enstrophy", dz < 0, f"(d enstrophy/dt {dz:.3e})")


def pipeline_semantics():
    g = Grid(32)
    model = model_for(g)
    wh = spectral.random_initial(g, jax.random.PRNGKey(10), batch=(1,))
    dev = jax.devices()[0]
    for mode in ("value", "operator"):
        got, _ = pipeline.run(g, model, wh, mode=mode, k=2, dt=0.01, n_outer=4, lag=0,
                              solver_device=dev, closure_device=dev)
        ref = schedule.advance(g, model, wh, mode=mode, k=2, dt=0.01, n_outer=4)
        err0 = float(jnp.max(jnp.abs(got - ref)) / jnp.max(jnp.abs(ref)))
        got1, _ = pipeline.run(g, model, wh, mode=mode, k=2, dt=0.01, n_outer=4, lag=1,
                               solver_device=dev, closure_device=dev)
        # manual lag 1: interval n uses the refresh of the state at the start of interval n - 1
        w, starts = wh, []
        for n in range(4):
            starts.append(w)
            held = schedule._refresh(g, model, starts[max(n - 1, 0)], mode, 0.17)
            w = pipeline.inner(g, w, held, mode, 2, 0.01, 0.17)
        err1 = float(jnp.max(jnp.abs(got1 - w)) / jnp.max(jnp.abs(w)))
        check(f"pipeline ({mode}): lag 0 = the scan schedule, lag 1 = the lagged loop",
              err0 < 1e-5 and err1 < 1e-5, f"({err0:.1e}, {err1:.1e})")



def pipeline3d_semantics():
    g = s3.Grid3(16, dealias="exp")
    model = split3d.init_model(jax.random.PRNGKey(54), hidden=4, blocks=1, rates=(1, 2))
    model["params"]["dec"]["w"] = 0.05 * jax.random.normal(jax.random.PRNGKey(55), model["params"]["dec"]["w"].shape)
    u = s3.random_initial(g, jax.random.PRNGKey(56), batch=(1,))
    dev = jax.devices()[0]
    got, _ = pipeline3d.run(g, model, u, k=2, dt=0.02, n_outer=3, lag=0, solver_device=dev, closure_device=dev)
    ref = split3d.advance(g, model, u, k=2, dt=0.02, n_steps=6)
    err0 = float(jnp.max(jnp.abs(got - ref)) / jnp.max(jnp.abs(ref)))
    got1, _ = pipeline3d.run(g, model, u, k=2, dt=0.02, n_outer=3, lag=1, solver_device=dev, closure_device=dev)
    # manual lag 1: block n uses the correction of the state at the start of block n - 1
    w, starts = u, []
    for n in range(3):
        starts.append(w)
        held = split3d.correction(g, model, starts[max(n - 1, 0)])
        w = pipeline3d.inner(g, w, held, 2, 0.02, split3d.base_of(model))
    err1 = float(jnp.max(jnp.abs(got1 - w)) / jnp.max(jnp.abs(w)))
    check("pipeline3d: lag 0 = the hold schedule, lag 1 = the lagged loop", err0 < 1e-5 and err1 < 1e-5,
          f"({err0:.1e}, {err1:.1e})")


def mixed_model3d():
    g = s3.Grid3(16, dealias="exp")
    u = s3.random_initial(g, jax.random.PRNGKey(57), batch=(1,))
    smag = s3.smagorinsky(g, 0.1)(u)
    no_grad = s3.mixed(g, 0.1, 0.0)(u)
    e0 = float(jnp.max(jnp.abs(no_grad - smag)) / jnp.max(jnp.abs(smag)))
    full = s3.mixed(g, 0.1)(u)
    # the gradient term transfers energy between scales: it changes the tendency but adds no net
    # mean momentum, and the mixed model still removes energy on a random field
    ke_rate = float(jnp.sum(jnp.real(jnp.conj(u) * full) * g.weight))
    check("mixed model: c_g = 0 is Smagorinsky; gradient term changes the tendency, zero-mean, dissipative",
          e0 < 1e-5 and float(jnp.max(jnp.abs(full - smag))) > 0 and float(jnp.max(jnp.abs(full[..., 0, 0, 0]))) < 1e-6
          and ke_rate < 0, f"({e0:.1e}, dE/dt {ke_rate:.2e})")


def mhd3d_physics():
    g = s3.Grid3(16)  # 2/3 rule: Galerkin truncation conserves the ideal invariants
    key = jax.random.PRNGKey(58)
    s = mhd3d.random_initial(g, key, batch=(1,))
    # b = 0: the MHD step is the Navier-Stokes step, and b stays 0
    u_only = mhd3d.join(mhd3d.fields(s)[0], jnp.zeros_like(mhd3d.fields(s)[0]))
    got = mhd3d.step(g, u_only, 0.01)
    ref = s3.step(g, mhd3d.fields(s)[0], 0.01)
    e_hydro = float(jnp.max(jnp.abs(mhd3d.fields(got)[0] - ref)) / jnp.max(jnp.abs(ref)))
    b_stays = float(jnp.max(jnp.abs(mhd3d.fields(got)[1])))
    # ideal, unforced: total energy and cross helicity conserved; both fields divergence-free
    gi = s3.Grid3(16, nu=0.0, power=0.0)
    x = s
    e0, h0 = sum(mhd3d.energies(gi, x)), mhd3d.cross_helicity(gi, x)
    for _ in range(20):
        x = mhd3d.step(gi, x, 0.002)
    e1, h1 = sum(mhd3d.energies(gi, x)), mhd3d.cross_helicity(gi, x)
    de = float(jnp.abs(e1 / e0 - 1.0)[0])
    dh = float(jnp.abs(h1 - h0)[0] / e0[0])
    div = max(float(s3.divergence_max(gi, mhd3d.fields(x)[0])), float(s3.divergence_max(gi, mhd3d.fields(x)[1])))
    # the classical closures remove total energy
    rates = []
    for kind, c in (("smag", 0.17), ("mixed", 0.17), ("cl", 1.0)):
        t = mhd3d.les_tendency(g, kind, c)(s)
        rates.append(float(jnp.sum(jnp.real(jnp.conj(s) * t) * g.weight)))
    check("MHD: b = 0 is Navier-Stokes; ideal energy and cross helicity conserved; fields divergence-free; "
          "LES closures dissipate", e_hydro < 1e-5 and b_stays < 1e-6 and de < 1e-4 and dh < 1e-4 and div < 1e-4
          and all(r < 0 for r in rates),
          f"(hydro {e_hydro:.1e}, b {b_stays:.1e}, dE {de:.1e}, dHc {dh:.1e}, div {div:.1e}, "
          f"rates {', '.join(f'{r:.1e}' for r in rates)})")

def tiles_semantics():
    g = Grid(32)
    model = model_for(g)
    wh = spectral.random_initial(g, jax.random.PRNGKey(11), batch=(2,))
    # a valid-padded patch with a halo of one cell per layer equals the dense periodic call
    feats = closure.features(g, wh, model["scale"])
    dense = closure.apply_cnn(model["params"], feats)
    idx = jnp.array([[0, 5, 15], [3, 7, 9]])
    halo = len(model["params"])
    patches = tiles.gather(feats, idx, 8, halo)
    out = closure.apply_cnn(model["params"], patches.reshape(6, *patches.shape[2:]),
                            padding="valid").reshape(2, 3, 8, 8, -1)
    want = tiles.gather(dense, idx, 8, 0)
    err = float(jnp.max(jnp.abs(out - want)) / jnp.max(jnp.abs(want)))
    # scatter of gathered centres is the identity
    back = tiles.scatter(jnp.zeros_like(dense), want, idx, 8)
    err2 = float(jnp.max(jnp.abs(tiles.gather(back, idx, 8, 0) - want)))
    # refreshing every tile every step is the every-step operator schedule
    got = tiles.advance(g, model, wh, m=16, t=8, age_max=1, dt=0.01, n_steps=4)
    ref = schedule.advance(g, model, wh, mode="operator", k=1, dt=0.01, n_outer=4)
    err3 = float(jnp.max(jnp.abs(got - ref)) / jnp.max(jnp.abs(ref)))
    check("tiles: valid patches = dense CNN, scatter inverts gather, full budget = B0",
          err < 1e-5 and err2 == 0.0 and err3 < 1e-5, f"({err:.1e}, {err2:.1e}, {err3:.1e})")


def tensor_closure_structure():
    g = Grid(32)
    params = closure.init_cnn(jax.random.PRNGKey(12), hidden=8, layers=3, out_ch=4)
    params[-1]["w"] = jax.random.normal(jax.random.PRNGKey(13), params[-1]["w"].shape) * 0.5
    model = {"params": params, "scale": {"u": 1.0, "w": 5.0, "k": 5e-3, "s": 0.1}}
    wh = spectral.random_initial(g, jax.random.PRNGKey(14), batch=(2,))
    coef = closure.coefficients(g, model, wh)
    w = spectral.to_real(g, wh)
    diss = coef.at[..., 3:].set(0.0)            # K only
    skew = coef.at[..., :3].set(0.0)            # curl s only
    z_diss = float(jnp.mean(w * spectral.to_real(g, closure.apply_operator(g, diss, wh))))
    z_skew = float(jnp.mean(w * spectral.to_real(g, closure.apply_operator(g, skew, wh))))
    t_skew = spectral.to_real(g, closure.apply_operator(g, skew, wh))
    scale = float(jnp.sqrt(jnp.mean(w**2) * jnp.mean(t_skew**2)))
    psd = bool(jnp.all(coef[..., 0] * coef[..., 2] - coef[..., 1] ** 2 >= -1e-12))
    check("tensor closure: K is PSD and removes enstrophy, curl s conserves it",
          psd and z_diss < 0 and abs(z_skew) < 1e-5 * scale,
          f"(dZ diss {z_diss:.2e}, dZ skew {z_skew:.1e} of {scale:.1e})")
    got = schedule.advance(g, model, wh, mode="operator", k=1, dt=0.01, n_outer=3)
    ref = wh
    for _ in range(3):
        c = closure.coefficients(g, model, ref)
        ref = spectral.step(g, ref, 0.01, lambda x, c=c: closure.apply_operator(g, c, x))
    tiles_full = tiles.advance(g, model, wh, m=16, t=8, age_max=1, dt=0.01, n_steps=3)
    e1 = float(jnp.max(jnp.abs(got - ref)) / jnp.max(jnp.abs(ref)))
    e2 = float(jnp.max(jnp.abs(tiles_full - ref)) / jnp.max(jnp.abs(ref)))
    check("tensor closure through the schedule and full-budget tiles = every-step", e1 < 1e-5 and e2 < 1e-5,
          f"({e1:.1e}, {e2:.1e})")


def finite_volume_structure():
    g = fv.FVGrid(32)
    w = spectral.to_real(Grid(32), spectral.random_initial(Grid(32), jax.random.PRNGKey(20), batch=(2,)))
    ux, vy = fv.face_velocities(g, w)
    div = float(jnp.max(jnp.abs(fv.divergence(g, ux, vy))))
    psi = fv.streamfunction(g, w)
    lap = (fv.sh(psi, 1, 0) + fv.sh(psi, -1, 0) + fv.sh(psi, 0, 1) + fv.sh(psi, 0, -1) - 4 * psi) / g.dx**2
    pois = float(jnp.max(jnp.abs(-lap - (w - w.mean(axis=(-2, -1), keepdims=True)))))
    check("FV: face velocities divergence-free, Poisson inverts the 5-point Laplacian",
          div < 1e-3 and pois < 1e-3 * float(jnp.max(jnp.abs(w))), f"(div {div:.1e}, poisson {pois:.1e})")
    zc = float(jnp.sum(w * fv.advection(g, ux, vy, *fv.interp_central(w))))
    zu = float(jnp.sum(w * fv.advection(g, ux, vy, *fv.interp_upwind3(w, ux, vy))))
    size = float(jnp.sum(jnp.abs(w * fv.advection(g, ux, vy, *fv.interp_central(w)))))
    check("FV: central advection conserves enstrophy, upwind3 removes it",
          abs(zc) < 1e-4 * size and zu < 0, f"(central {zc:.1e} of {size:.1e}, upwind3 {zu:.2e})")
    params = closure.init_cnn(jax.random.PRNGKey(21), hidden=8, layers=3, out_ch=fv.N_LI)
    model = {"params": params, "scale": {"u": 1.0, "w": 5.0}}
    a = fv.advance(g, model, w, mode="li", k=1, dt=0.01, n_outer=3)
    b = fv.advance(g, model, w, mode="central", k=1, dt=0.01, n_outer=3)
    c = fv.advance(g, model, w, mode="li_value", k=2, dt=0.01, n_outer=2)
    e1 = float(jnp.max(jnp.abs(a - b)))
    e2 = float(jnp.max(jnp.abs(c - b[..., :, :]) * 0 + jnp.abs(
        fv.advance(g, model, w, mode="central", k=1, dt=0.01, n_outer=4) - c)))
    check("FV: untrained learned interpolation is central (operator and value holds)",
          e1 < 1e-5 and e2 < 1e-5, f"({e1:.1e}, {e2:.1e})")
    params[-1]["w"] = jax.random.normal(jax.random.PRNGKey(22), params[-1]["w"].shape)
    coef = fv.li_coefficients(g, model, w)
    const = jnp.full_like(w, 3.0)
    wx, wy = fv.interp_li(coef, const)
    cons = float(jnp.max(jnp.abs(wx - 3.0)) + jnp.max(jnp.abs(wy - 3.0)))
    m0 = w.mean(axis=(-2, -1))
    m1 = fv.advance(g, model, w, mode="li", k=2, dt=0.01, n_outer=3).mean(axis=(-2, -1))
    drift = float(jnp.max(jnp.abs(m1 - m0)))
    check("FV: learned stencils are consistent and mean vorticity is conserved",
          cons < 1e-5 and drift < 1e-4, f"(constant field error {cons:.1e}, mean drift {drift:.1e})")


def finite_volume_upwind_base():
    g = fv.FVGrid(32)
    w = spectral.to_real(Grid(32), spectral.random_initial(Grid(32), jax.random.PRNGKey(23), batch=(2,)))
    params = closure.init_cnn(jax.random.PRNGKey(24), hidden=8, layers=3, out_ch=fv.N_LI)
    model = {"params": params, "scale": {"u": 1.0, "w": 5.0, "bound": 0.25},
             "static": closure.Static(base="upwind3")}
    a = fv.advance(g, model, w, mode="li", k=2, dt=0.01, n_outer=2)
    b = fv.advance(g, model, w, mode="li_value", k=2, dt=0.01, n_outer=2)
    c = fv.advance(g, None, w, mode="upwind3", k=1, dt=0.01, n_outer=4)
    e1, e2 = float(jnp.max(jnp.abs(a - c))), float(jnp.max(jnp.abs(b - c)))
    check("FV: untrained correction on the upwind base is upwind3 (operator and value holds)",
          e1 < 1e-5 and e2 < 1e-5, f"({e1:.1e}, {e2:.1e})")


def finite_volume_advected_hold():
    g = fv.FVGrid(32)
    w = spectral.to_real(Grid(32), spectral.random_initial(Grid(32), jax.random.PRNGKey(25), batch=(2,)))
    params = closure.init_cnn(jax.random.PRNGKey(26), hidden=8, layers=3, out_ch=fv.N_LI)
    params[-1]["w"] = jax.random.normal(jax.random.PRNGKey(27), params[-1]["w"].shape) * 0.3
    model = {"params": params, "scale": {"u": 1.0, "w": 5.0, "bound": 0.25},
             "static": closure.Static(base="upwind3")}
    a = fv.advance(g, model, w, mode="li_adv", k=1, dt=0.01, n_outer=3)
    b = fv.advance(g, model, w, mode="li", k=1, dt=0.01, n_outer=3)
    c = fv.advance(g, model, w, mode="li_value_adv", k=1, dt=0.01, n_outer=3)
    d = fv.advance(g, model, w, mode="li_value", k=1, dt=0.01, n_outer=3)
    e1, e2 = float(jnp.max(jnp.abs(a - b))), float(jnp.max(jnp.abs(c - d)))
    f = jax.random.normal(jax.random.PRNGKey(28), (2, 32, 32, 3))
    still = fv.advect_cells(g, f, jnp.zeros((2, 32, 32)), 0.01)
    e3 = float(jnp.max(jnp.abs(still - f)))
    check("FV: advected holds at k = 1 are the plain holds; zero flow carries nothing",
          e1 < 1e-5 and e2 < 1e-5 and e3 < 1e-6, f"({e1:.1e}, {e2:.1e}, {e3:.1e})")


def finite_volume_student_residual():
    g = fv.FVGrid(32)
    w = spectral.to_real(Grid(32), spectral.random_initial(Grid(32), jax.random.PRNGKey(29), batch=(2,)))
    big = closure.init_cnn(jax.random.PRNGKey(30), hidden=8, layers=3, out_ch=fv.N_LI)
    big[-1]["w"] = jax.random.normal(jax.random.PRNGKey(31), big[-1]["w"].shape) * 0.3
    small = closure.init_cnn(jax.random.PRNGKey(32), hidden=4, layers=2, out_ch=fv.N_LI)
    small[-1]["w"] = jax.random.normal(jax.random.PRNGKey(33), small[-1]["w"].shape) * 0.3
    model = {"params": big, "student": small, "scale": {"u": 1.0, "w": 5.0, "bound": 0.25},
             "static": closure.Static(base="upwind3")}
    a = fv.advance(g, model, w, mode="li_res", k=1, dt=0.01, n_outer=3)
    b = fv.advance(g, model, w, mode="li", k=1, dt=0.01, n_outer=3)
    e1 = float(jnp.max(jnp.abs(a - b)))
    check("FV: the residual hold at k = 1 is the teacher every step", e1 < 1e-4, f"({e1:.1e})")


def finite_volume_certified():
    g = fv.FVGrid(32)
    w = spectral.to_real(Grid(32), spectral.random_initial(Grid(32), jax.random.PRNGKey(34), batch=(2,)))
    params = closure.init_cnn(jax.random.PRNGKey(35), hidden=8, layers=3, out_ch=fv.N_LI)
    params[-1]["w"] = jax.random.normal(jax.random.PRNGKey(36), params[-1]["w"].shape) * 0.3
    model = {"params": params, "scale": {"u": 1.0, "w": 5.0, "bound": 0.25},
             "static": closure.Static(base="upwind3")}
    lax_ = fv.cert_advance(g, model, w, k=4, dt=0.01, e_star=1e9, n_outer=2)   # never rolls back
    plain = fv.advance(g, model, w, mode="li", k=4, dt=0.01, n_outer=2)
    tight = fv.cert_advance(g, model, w, k=4, dt=0.01, e_star=-1.0, n_outer=1)  # always rolls back
    halves = fv.advance(g, model, w, mode="li", k=2, dt=0.01, n_outer=2)
    e1, e2 = float(jnp.max(jnp.abs(lax_ - plain))), float(jnp.max(jnp.abs(tight - halves)))
    check("FV certified: no violation = the plain hold; always violating = two halves",
          e1 < 1e-5 and e2 < 1e-5, f"({e1:.1e}, {e2:.1e})")


def finite_volume_cgrid():
    g, ga = fv.FVGrid(32, velocity="cgrid"), fv.FVGrid(32)
    w = spectral.to_real(Grid(32), spectral.random_initial(Grid(32), jax.random.PRNGKey(37), batch=(2,)))
    ux, vy = fv.face_velocities(g, w)
    div = float(jnp.max(jnp.abs(fv.divergence(g, ux, vy))))
    # one Fourier mode psi = sin(3x) sin(2y): exact face velocities, both schemes
    h = g.dx
    xc = (np.arange(32) + 0.5) * h
    X, Y = np.meshgrid(xc, xc, indexing="ij")
    psi = np.sin(3 * X) * np.sin(2 * Y)
    omega = jnp.asarray(13.0 * psi, jnp.float32)[None]  # omega = -lap psi = (9 + 4) psi
    u_true = 2 * np.sin(3 * (X + h / 2)) * np.cos(2 * Y)  # dpsi/dy at the x-faces
    errs = {}
    for name, grid in (("cgrid", g), ("avg", ga)):
        u, _ = fv.face_velocities(grid, omega)
        errs[name] = float(jnp.max(jnp.abs(u[0] - u_true)) / np.max(np.abs(u_true)))
    params = closure.init_cnn(jax.random.PRNGKey(38), hidden=8, layers=3, out_ch=fv.n_li(5))
    params[-1]["w"] = jax.random.normal(jax.random.PRNGKey(39), params[-1]["w"].shape)
    model = {"params": params, "scale": {"u": 1.0, "w": 5.0, "bound": 1.0},
             "static": closure.Static(base="upwind3")}
    coef = fv.li_coefficients(g, model, w)
    wx, wy = fv.interp_li(coef, jnp.full_like(w, 2.0), ux, vy, "upwind3")
    cons = float(jnp.max(jnp.abs(wx - 2.0)) + jnp.max(jnp.abs(wy - 2.0)))
    check("FV C-grid: divergence-free, more accurate than avg; width-5 stencils consistent",
          div < 1e-3 and errs["cgrid"] < 0.5 * errs["avg"] and cons < 1e-5,
          f"(div {div:.1e}, u error cgrid {errs['cgrid']:.3f} vs avg {errs['avg']:.3f}, const {cons:.1e})")


def exp_filter_grid():
    g = Grid(32, dealias="exp")
    f = g.exp_filter
    low, nyq = float(f[2, 2]), float(np.exp(-6.0 * (15 / 16) ** 32))
    near = float(f[15, 0])
    check("exp filter: band to n/2 - 1, ~1 at low k, exp(-6 (k/(n/2))^32) at the edge",
          g.kmax == 15 and abs(low - 1) < 1e-6 and abs(near - nyq) < 1e-6 and float(g.mask[16, 0]) == 0.0,
          f"(kmax {g.kmax}, f(2,2) {low:.6f}, f(15,0) {near:.4f})")


def split_operator():
    g = Grid(32, dealias="exp")
    w = spectral.truncate(Grid(32), g, spectral.random_initial(Grid(32), jax.random.PRNGKey(40), batch=(2,)))
    model = split.init_model(jax.random.PRNGKey(41), hidden=8, blocks=1, rates=(1, 2))
    plain = schedule.advance(g, None, w, mode="none", k=1, dt=0.01, n_outer=6)
    e0 = float(jnp.max(jnp.abs(split.advance(g, model, w, k=1, dt=0.01, n_steps=6) - plain)))
    dec = model["params"]["dec"]
    dec["w"] = 0.3 * jax.random.normal(jax.random.PRNGKey(42), dec["w"].shape)
    mean = float(jnp.max(jnp.abs(split.correction(g, model, w)[..., 0, 0])))
    manual, c = w, None
    for i in range(6):
        c = split.correction(g, model, manual) if i % 3 == 0 else c
        manual = split.split_step(g, manual, c, 0.01)
    e1 = float(jnp.max(jnp.abs(split.advance(g, model, w, k=3, dt=0.01, n_steps=6) - manual))
               / jnp.max(jnp.abs(manual)))  # eager against compiled: relative
    rec = split.rollout(g, model, w, k=4, dt=0.01, n_records=3, record_every=2)
    e2 = float(jnp.max(jnp.abs(rec[-1] - split.advance(g, model, w, k=4, dt=0.01, n_steps=6))))
    e3 = float(jnp.max(jnp.abs(split.cert_advance(g, model, w, k=4, dt=0.01, e_star=1e9, n_steps=8)
                               - split.advance(g, model, w, k=4, dt=0.01, n_steps=8))))
    e4 = float(jnp.max(jnp.abs(split.cert_advance(g, model, w, k=4, dt=0.01, e_star=-1.0, n_steps=4)
                               - split.advance(g, model, w, k=2, dt=0.01, n_steps=4))))
    st, n_rb, _ = split.cert_rollout(g, model, w, k=4, dt=0.01, e_star=1e9, n_records=3, record_every=2)
    e5 = float(jnp.max(jnp.abs(st - rec))) + float(n_rb)
    ref = split.advance(g, model, w, k=1, dt=0.01, n_steps=4)
    e6 = float(jnp.max(jnp.abs(split.advance(g, model, w, mode="advected", k=1, dt=0.01, n_steps=4) - ref))
               / jnp.max(jnp.abs(ref)))
    errs = (e0, e1, e2, e3, e4, e5, e6)
    check("split operator: untrained = filtered spectral; zero-mean correction; holds span records; "
          "certified = hold / two halves; advected k = 1 = every step",
          max(errs) < 1e-5 and mean < 1e-6, "(" + ", ".join(f"{e:.1e}" for e in errs) + f", mean {mean:.1e})")


def spectral3d_physics():
    # inviscid, unforced: the 2/3-rule Galerkin truncation conserves energy and keeps div u = 0
    g = s3.Grid3(16, nu=0.0, power=0.0)
    u = s3.random_initial(g, jax.random.PRNGKey(50), batch=(2,))
    e0 = s3.energy_of(g, u)
    v = u
    for _ in range(10):
        v = s3.step(g, v, 0.01)
    drift = float(jnp.max(jnp.abs(s3.energy_of(g, v) / e0 - 1)))
    div = float(s3.divergence_max(g, v)) / float(s3.max_speed(g, v))
    # a shear wave u_x = sin z is steady for Euler and decays as exp(-nu t)
    gv = s3.Grid3(16, nu=0.05, power=0.0)
    x = np.arange(16) * 2 * np.pi / 16
    field = np.zeros((1, 3, 16, 16, 16), np.float32)
    field[0, 0] = np.sin(x)[None, None, :]
    wh = s3.to_hat(gv, jnp.asarray(field))
    for _ in range(20):
        wh = s3.step(gv, wh, 0.01)
    decay = float(jnp.max(jnp.abs(s3.to_real(gv, wh) - jnp.asarray(field) * np.exp(-0.05 * 0.2))))
    # the forcing injects exactly the set power
    gf = s3.Grid3(16, nu=0.0, power=0.3)
    f = s3.forcing(gf, u)
    inj = float(jnp.max(jnp.abs(jnp.sum(jnp.real(f * jnp.conj(u)) * gf.weight, axis=(-4, -3, -2, -1))
                                / 16.0**6 - 0.3)))
    # truncation inverts band-limited zero padding (32 -> 16)
    gc, gfine = s3.Grid3(16), s3.Grid3(32)
    small = s3.random_initial(gc, jax.random.PRNGKey(51))
    padded = jnp.zeros((3, 32, 32, 17), small.dtype)
    kc = gc.kmax
    for xc, xf in ((slice(0, kc + 1), slice(0, kc + 1)), (slice(16 - kc, 16), slice(32 - kc, 32))):
        for yc, yf in ((slice(0, kc + 1), slice(0, kc + 1)), (slice(16 - kc, 16), slice(32 - kc, 32))):
            padded = padded.at[:, xf, yf, : kc + 1].set(small[:, xc, yc, : kc + 1] * 8.0)
    back = float(jnp.max(jnp.abs(s3.truncate(gfine, gc, padded) - small)) / jnp.max(jnp.abs(small)))
    check("3D spectral: inviscid energy conserved, div u = 0, shear wave decays at exp(-nu t), "
          "forcing injects the set power, truncation inverts zero padding",
          drift < 1e-4 and div < 1e-5 and decay < 1e-5 and inj < 1e-4 and back < 1e-6,
          f"(energy drift {drift:.1e}, div {div:.1e}, decay {decay:.1e}, power {inj:.1e}, trunc {back:.1e})")


def split3d_operator():
    for arch in ("epd", "dw"):
        _split3d_case(arch)


def _split3d_case(arch: str):
    g = s3.Grid3(12, dealias="exp")
    u = s3.truncate(s3.Grid3(12), g, s3.random_initial(s3.Grid3(12), jax.random.PRNGKey(52), batch=(2,)))
    model = split3d.init_model(jax.random.PRNGKey(53), arch=arch, hidden=4, blocks=1, rates=(1, 2))
    plain = u
    for _ in range(3):
        plain = s3.step(g, plain, 0.02)
    e0 = float(jnp.max(jnp.abs(split3d.advance(g, model, u, k=1, dt=0.02, n_steps=3) - plain))
               / jnp.max(jnp.abs(plain)))  # eager against compiled: relative
    dec = model["params"]["dec"]
    dec["w"] = 0.3 * jax.random.normal(jax.random.PRNGKey(54), dec["w"].shape)
    c = split3d.correction(g, model, u)
    div = float(s3.divergence_max(g, c)) / float(jnp.max(jnp.abs(s3.to_real(g, c))))
    manual, held = u, None
    for i in range(4):
        held = split3d.correction(g, model, manual) if i % 2 == 0 else held
        manual = split3d.split_step(g, manual, held, 0.02)
    e1 = float(jnp.max(jnp.abs(split3d.advance(g, model, u, k=2, dt=0.02, n_steps=4) - manual))
               / jnp.max(jnp.abs(manual)))
    e2 = float(jnp.max(jnp.abs(split3d.cert_advance(g, model, u, k=2, dt=0.02, e_star=1e9, n_steps=4)
                               - split3d.advance(g, model, u, k=2, dt=0.02, n_steps=4))))
    e3 = float(jnp.max(jnp.abs(split3d.cert_advance(g, model, u, k=2, dt=0.02, e_star=-1.0, n_steps=2)
                               - split3d.advance(g, model, u, k=1, dt=0.02, n_steps=2))))
    rec = split3d.rollout(g, model, u, k=2, dt=0.02, n_records=3, record_every=1)
    e4 = float(jnp.max(jnp.abs(rec[-1] - split3d.advance(g, model, u, k=2, dt=0.02, n_steps=3))))
    errs = (e0, e1, e2, e3, e4)
    check(f"3D split operator ({arch}): untrained = filtered spectral; correction divergence-free; holds, "
          "certificate and records consistent", max(errs) < 1e-5 and div < 1e-5,
          "(" + ", ".join(f"{e:.1e}" for e in errs) + f", div {div:.1e})")


def pad_and_les3d():
    # pad is the inverse of truncate on the coarse band, in 2D and 3D
    g2c, g2f = Grid(16, dealias="exp"), Grid(32, dealias="exp")
    w = spectral.truncate(Grid(32), g2c, spectral.random_initial(Grid(32), jax.random.PRNGKey(60)))
    back2 = float(jnp.max(jnp.abs(spectral.truncate(g2f, g2c, spectral.pad(g2c, g2f, w)) / g2c.exp_filter
                                  * (g2c.exp_filter > 0) - w * (g2c.exp_filter > 0))) / jnp.max(jnp.abs(w)))
    g3c, g3f = s3.Grid3(12), s3.Grid3(24)
    u = s3.random_initial(g3c, jax.random.PRNGKey(61), batch=(2,))
    back3 = float(jnp.max(jnp.abs(s3.truncate(g3f, g3c, s3.pad(g3c, g3f, u)) - u)) / jnp.max(jnp.abs(u)))
    # the classical LES closures remove energy, and a step with either stays finite and solenoidal
    g = s3.Grid3(16, dealias="exp")
    u = s3.random_initial(g, jax.random.PRNGKey(62), batch=(2,))
    rates, ok = [], True
    for kind, c in (("cl", 1.0), ("smag", 0.17)):
        f = s3.project(g, s3.les_tendency(g, kind, c)(u) * g.mask)
        rates.append(float(jnp.max(jnp.sum(jnp.real(f * jnp.conj(u)) * g.weight, axis=(-4, -3, -2, -1)))))
        v = s3.step(g, u, 0.02, s3.les_tendency(g, kind, c))
        ok &= bool(jnp.all(jnp.isfinite(v.real))) and float(s3.divergence_max(g, v) / s3.max_speed(g, v)) < 1e-5
    check("pad inverts truncate (2D, 3D); Chollet-Lesieur and Smagorinsky dissipate, steps stay solenoidal",
          back2 < 1e-5 and back3 < 1e-6 and max(rates) < 0 and ok,
          f"(pad 2D {back2:.1e}, 3D {back3:.1e}, dE/dt CL {rates[0]:.2e}, Smag {rates[1]:.2e}, ok {ok})")


def tracer_schemes():
    from frost.cfd import tracers

    # face velocities from a vorticity field are discretely divergence-free
    flow, g = Grid(16, dealias="exp"), Grid(32, dealias="exp")
    w = spectral.truncate(Grid(32), flow, spectral.random_initial(Grid(32), jax.random.PRNGKey(70), batch=(2,)))
    ux, vy = tracers.face_velocity(flow, g, w)
    div = float(jnp.max(jnp.abs((ux - jnp.roll(ux, 1, -2)) + (vy - jnp.roll(vy, 1, -1)))) / jnp.max(jnp.abs(ux)))
    # cell averages round-trip, and every scheme conserves mass; van Leer stays non-negative
    c_hat = tracers.random_tracer(g, jax.random.PRNGKey(71), batch=(2,))
    ca = tracers.CellAverage(g)
    trip = float(jnp.max(jnp.abs(ca.to_points(ca.from_points(c_hat)) - c_hat)) / jnp.max(jnp.abs(c_hat)))
    c0 = ca.from_points(c_hat)
    dt = 0.3 * g.dx / float(jnp.max(jnp.abs(jnp.stack([ux, vy]))))
    mass, low = [], {}
    for scheme in tracers.FV_SCHEMES:
        c = c0
        for _ in range(20):
            c = tracers.fv_step(scheme, g.dx, c, ux, vy, dt, 1e-3)
        mass.append(float(jnp.max(jnp.abs(c.mean((-2, -1)) - c0.mean((-2, -1))))))
        low[scheme] = float(c.min())
    u, v = tracers.point_velocity(flow, g, w)
    cs = c_hat
    for _ in range(20):
        cs = tracers.spectral_step(g, cs, u, v, dt, 1e-3)
    mass.append(float(jnp.max(jnp.abs(cs[..., 0, 0] - c_hat[..., 0, 0]))) / g.n**2)
    # convergence on a smooth field. Both finite-volume schemes are dimension by dimension (the flux
    # is the face-mean velocity times a face-mean tracer), so second order in 2D with a variable
    # velocity, as in Athena++ or FV3; WENO-Z's constant is far smaller than van Leer's
    ref_g, t_end = Grid(256, dealias="exp"), 0.2
    uf, vf = tracers.point_velocity(flow, ref_g, w)
    cr = spectral.pad(g, ref_g, c_hat)
    for _ in range(128):
        cr = tracers.spectral_step(ref_g, cr, uf, vf, t_end / 128, 0.0)
    errs = {}
    for scheme in ("vanleer", "weno5z"):
        for n in (64, 128):
            gn = Grid(n, dealias="exp")
            uxn, vyn = tracers.face_velocity(flow, gn, w)
            can = tracers.CellAverage(gn)
            c = can.from_points(spectral.pad(g, gn, c_hat))
            steps = 2 * n
            for _ in range(steps):
                c = tracers.fv_step(scheme, gn.dx, c, uxn, vyn, t_end / steps, 0.0)
            exact = can.from_points(tracers.truncate_sharp(ref_g, n, n // 2 - 1, cr))
            errs[(scheme, n)] = float(jnp.sqrt(jnp.mean((c - exact) ** 2)))
    order = {s: np.log2(errs[(s, 64)] / errs[(s, 128)]) for s in ("vanleer", "weno5z")}
    gain = errs[("vanleer", 128)] / errs[("weno5z", 128)]
    check("tracers: divergence-free faces, cell-average round trip, mass conserved, van Leer and limited WENO-Z positive, "
          "finite volume converges (WENO-Z well below van Leer)",
          div < 1e-5 and trip < 1e-5 and max(mass) < 1e-5 and min(low["vanleer"], low["weno5z_pp"]) >= -1e-6
          and min(order.values()) > 1.3
          and gain > 3.0,
          f"(div {div:.1e}, trip {trip:.1e}, mass {max(mass):.1e}, min vanleer {low['vanleer']:.2e}, "
          f"weno5z {low['weno5z']:.2e}, weno5z_pp {low['weno5z_pp']:.2e}, "
          f"order vanleer {order['vanleer']:.2f}, weno5z {order['weno5z']:.2f}, error ratio at 128 {gain:.1f})")


def stochastic_closure3d():
    """A stochastic closure (split3d.add_noise_head) with a vanishing amplitude reproduces the deterministic rollout; with
    an amplitude it differs, its draws change from refresh to refresh, and a hold keeps one draw for k steps."""
    g = s3.Grid3(12, dealias="exp")
    det = split3d.init_model(jax.random.PRNGKey(61), hidden=4, blocks=1, rates=(1, 2))
    u = s3.truncate(s3.Grid3(12), g, s3.random_initial(s3.Grid3(12), jax.random.PRNGKey(62), batch=(2,)))
    det = {**det, "params": {**det["params"], "dec": {**det["params"]["dec"],
                                                      "w": 0.05 * jnp.ones_like(det["params"]["dec"]["w"])}}}
    quiet = split3d.add_noise_head(det, jax.random.PRNGKey(63), sigma_init=-60.0)
    loud = split3d.add_noise_head(det, jax.random.PRNGKey(63), sigma_init=2.0)
    ref = split3d.advance(g, det, u, k=2, dt=0.02, n_steps=4)
    e_quiet = float(jnp.max(jnp.abs(split3d.advance(g, quiet, u, k=2, dt=0.02, n_steps=4) - ref)))
    e_loud = float(jnp.max(jnp.abs(split3d.advance(g, loud, u, k=2, dt=0.02, n_steps=4) - ref)))
    c0, c1 = (split3d.correction(g, loud, u, step=i) for i in (0, 1))
    c_mean = split3d.correction(g, loud, u)
    fresh = float(jnp.max(jnp.abs(c0 - c1)))
    mean_is_det = float(jnp.max(jnp.abs(c_mean - split3d.correction(g, det, u))))
    div = float(s3.divergence_max(g, c0))
    check("3D stochastic closure: zero amplitude = deterministic, an amplitude changes the rollout, draws differ by "
          "refresh, the mean head without a step is the deterministic correction, noise divergence-free",
          e_quiet < 1e-6 and e_loud > 1e-4 and fresh > 1e-4 and mean_is_det < 1e-7 and div < 1e-4,
          f"(quiet {e_quiet:.1e}, loud {e_loud:.1e}, refresh {fresh:.1e}, mean {mean_is_det:.1e}, div {div:.1e})")

if __name__ == "__main__":
    for fn in (single_mode_decays_exactly, inviscid_invariants, spectrum_sums_to_energy,
               truncation_is_a_filter, flux_closure_conserves, operator_and_value_agree_at_one_stage,
               hold_with_k1_matches_manual_loop, held_value_is_held, advection_transports,
               smagorinsky_dissipates, pipeline_semantics, pipeline3d_semantics, mixed_model3d, mhd3d_physics,
               tiles_semantics, tensor_closure_structure, finite_volume_structure,
               finite_volume_upwind_base, finite_volume_advected_hold,
               finite_volume_student_residual, finite_volume_certified,
               finite_volume_cgrid, exp_filter_grid, split_operator,
               spectral3d_physics, split3d_operator, stochastic_closure3d, pad_and_les3d, tracer_schemes):
        fn()
    if FAILED:
        print(f"FAILED {len(FAILED)}: {', '.join(FAILED)}")
        sys.exit(1)
    print(f"ok {len(PASSED)} checks")
