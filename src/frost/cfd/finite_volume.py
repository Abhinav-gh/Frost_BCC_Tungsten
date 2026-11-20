"""Finite-volume Kolmogorov flow with learned interpolation (workload W1, second form).

The spectral closure of ``frost.cfd.spectral`` was stable but weak. DS 128^2 beat it
on both cost and accuracy (test 11, 2026-10-01), as Dresdner et al. (2022) found for
learned corrections to spectral solvers. Kochkov et al. (2021) got 8-10x in
resolution on a *finite-volume* solver by learning the interpolation inside the
advection operator. This module is that design, in vorticity form, our own code:

- omega lives at cell centres on the periodic box [0, 2 pi)^2;
- psi solves the 5-point Poisson equation exactly (FFT of its symbol);
- the face velocities, averaged central differences of psi, are discretely
  divergence-free, so flux-form advection conserves mean vorticity;
- advection: -div(u omega_face), with omega_face interpolated from the cells;
- diffusion: the 5-point Laplacian; drag; Kolmogorov forcing; SSP-RK3 in time.

Interpolations:
- ``upwind3``: third-order upwind-biased, the classical baseline;
- ``central``: second order, the base the learned stencil corrects;
- ``li``: central + d, where a CNN gives each face a 4x3 stencil correction d. Each
  d is bounded (0.5 tanh) and projected to zero sum, so the interpolation stays
  consistent (sum of weights = 1), as in Kochkov's constrained stencils.

The learned coefficients are **operator-consumed**: held, they are applied to the
fresh state at every stage and every step. Their value-consumed counterpart (the
held tendency of the learned flux correction, added to fresh central advection) is
the ``li_value`` hold.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.ndimage import map_coordinates

from frost.cfd import closure

TWO_PI = 2.0 * np.pi
def stencils(width: int = 3):
    """Learned-stencil offsets: 4 cells along each face's normal, ``width`` across it."""
    across = range(-(width // 2), width // 2 + 1)
    x = [(a, b) for a in (-1, 0, 1, 2) for b in across]  # x-face (i+1/2, j)
    y = [(a, b) for a in across for b in (-1, 0, 1, 2)]  # y-face (i, j+1/2)
    return x, y


def n_li(width: int = 3) -> int:
    return 8 * width


X_STENCIL, Y_STENCIL = stencils(3)
N_LI = n_li(3)


@dataclass(frozen=True)
class FVGrid:
    """``velocity`` sets how face velocities come from omega:
    - "avg": averaged two-cell central differences of a 5-point-Poisson psi at centres;
    - "cgrid": psi solved spectrally, moved to cell corners by an exact half-cell Fourier
      shift, then differenced across one cell.

    Both are discretely divergence-free. The C-grid's phase error at the highest resolved
    wavenumber of 64^2 (k = 21) is ~18% against ~57% for "avg", an error the learned
    omega interpolation cannot see (2026-10-02).
    """

    n: int
    nu: float = 1e-3
    alpha: float = 0.1
    k_force: int = 4
    velocity: str = "avg"

    @property
    def dx(self) -> float:
        return TWO_PI / self.n

    @cached_property
    def lap_symbol(self) -> np.ndarray:
        k = np.fft.fftfreq(self.n) * self.n
        q = np.fft.rfftfreq(self.n) * self.n
        h = self.dx
        sym = (2 * np.cos(k[:, None] * h) - 2 + 2 * np.cos(q[None, :] * h) - 2) / h**2
        inv = np.zeros_like(sym)
        np.divide(-1.0, sym, out=inv, where=sym != 0)  # psi_hat = -omega_hat / lambda
        return inv.astype(np.float32)

    @cached_property
    def corner_psi_symbol(self) -> np.ndarray:
        """omega_hat -> psi_hat at the corners: the exact 1/|k|^2 and a half-cell shift in x, y."""
        k = (np.fft.fftfreq(self.n) * self.n)[:, None]
        q = (np.fft.rfftfreq(self.n) * self.n)[None, :]
        k2 = k**2 + q**2
        inv = np.zeros_like(k2)
        np.divide(1.0, k2, out=inv, where=k2 > 0)
        return (inv * np.exp(1j * (k + q) * self.dx / 2)).astype(np.complex64)

    @cached_property
    def forcing(self) -> np.ndarray:
        y = (np.arange(self.n) + 0.5) * self.dx
        return (-self.k_force * np.cos(self.k_force * y))[None, :].repeat(self.n, 0).astype(np.float32)


def sh(f, a: int, b: int):
    """f(i + a, j + b), periodic."""
    return jnp.roll(f, shift=(-a, -b), axis=(-2, -1))


def streamfunction(g: FVGrid, w):
    return jnp.fft.irfft2(jnp.fft.rfft2(w) * g.lap_symbol, s=(g.n, g.n))


def face_velocities(g: FVGrid, w):
    """u on x-faces (i+1/2, j) and v on y-faces (i, j+1/2); discretely divergence-free."""
    if g.velocity == "cgrid":
        # psi_c[i, j] = psi at the corner ((i+1)h, (j+1)h); one-cell differences across faces
        psi_c = jnp.fft.irfft2(jnp.fft.rfft2(w) * g.corner_psi_symbol, s=(g.n, g.n))
        ux = (psi_c - sh(psi_c, 0, -1)) / g.dx
        vy = -(psi_c - sh(psi_c, -1, 0)) / g.dx
        return ux, vy
    psi = streamfunction(g, w)
    q = 4.0 * g.dx
    ux = ((sh(psi, 0, 1) - sh(psi, 0, -1)) + (sh(psi, 1, 1) - sh(psi, 1, -1))) / q
    vy = -((sh(psi, 1, 0) - sh(psi, -1, 0)) + (sh(psi, 1, 1) - sh(psi, -1, 1))) / q
    return ux, vy


def divergence(g: FVGrid, ux, vy):
    return (ux - sh(ux, -1, 0) + vy - sh(vy, 0, -1)) / g.dx


def interp_central(w):
    return 0.5 * (w + sh(w, 1, 0)), 0.5 * (w + sh(w, 0, 1))


def interp_upwind3(w, ux, vy):
    xp = (-sh(w, -1, 0) + 5 * w + 2 * sh(w, 1, 0)) / 6
    xm = (2 * w + 5 * sh(w, 1, 0) - sh(w, 2, 0)) / 6
    yp = (-sh(w, 0, -1) + 5 * w + 2 * sh(w, 0, 1)) / 6
    ym = (2 * w + 5 * sh(w, 0, 1) - sh(w, 0, 2)) / 6
    return jnp.where(ux >= 0, xp, xm), jnp.where(vy >= 0, yp, ym)


# ------------------------------------------------------------------ learned interpolation
def features(g: FVGrid, w, scale: dict):
    ux, vy = face_velocities(g, w)
    uc = 0.5 * (ux + sh(ux, -1, 0))
    vc = 0.5 * (vy + sh(vy, 0, -1))
    return jnp.stack([uc / scale["u"], vc / scale["u"], w / scale["w"]], axis=-1)


def li_from_raw(raw, bound: float = 0.5):
    """Network output (..., 8w) -> zero-sum, bounded stencil corrections (..., 8w)."""
    d = bound * jnp.tanh(raw)
    m = raw.shape[-1] // 2
    dx_ = d[..., :m] - d[..., :m].mean(-1, keepdims=True)
    dy_ = d[..., m:] - d[..., m:].mean(-1, keepdims=True)
    return jnp.concatenate([dx_, dy_], axis=-1)


def li_coefficients(g: FVGrid, model: dict, w):
    """The expensive part: one CNN call. (B, n, n) -> (B, n, n, 24)."""
    return li_from_raw(closure.apply_cnn(model["params"], features(g, w, model["scale"])),
                       model["scale"].get("bound", 0.5))


def base_of(held_model_base: str, w, ux, vy):
    return interp_upwind3(w, ux, vy) if held_model_base == "upwind3" else interp_central(w)


def interp_li(coef, w, ux=None, vy=None, base: str = "central"):
    """The base interpolation plus the held learned correction, on the given (fresh) state.

    ``base="upwind3"`` puts the correction on a stable, dissipative scheme: the
    multirate shape (a stable cheap student plus a small learned residual). On the
    unstable central base the correction was itself the stabiliser, and every hold
    broke it (2026-10-02).
    """
    wx, wy = base_of(base, w, ux, vy)
    m = coef.shape[-1] // 2
    xs, ys = stencils(m // 4)  # the stencil width is the network's output size / 8
    for s, (a, b) in enumerate(xs):
        wx = wx + coef[..., s] * sh(w, a, b)
    for s, (a, b) in enumerate(ys):
        wy = wy + coef[..., m + s] * sh(w, a, b)
    return wx, wy


def advection(g: FVGrid, ux, vy, wx, wy):
    fx, fy = ux * wx, vy * wy
    return -((fx - sh(fx, -1, 0)) + (fy - sh(fy, 0, -1))) / g.dx


def rhs(g: FVGrid, w, mode: str, held, base: str = "central"):
    ux, vy = face_velocities(g, w)
    if mode == "upwind3":
        adv = advection(g, ux, vy, *interp_upwind3(w, ux, vy))
    elif mode == "central":
        adv = advection(g, ux, vy, *interp_central(w))
    elif mode == "li":
        adv = advection(g, ux, vy, *interp_li(held, w, ux, vy, base))
    elif mode == "li_value":
        adv = advection(g, ux, vy, *base_of(base, w, ux, vy)) + held
    else:
        raise ValueError(mode)
    lap = (sh(w, 1, 0) + sh(w, -1, 0) + sh(w, 0, 1) + sh(w, 0, -1) - 4 * w) / g.dx**2
    return adv + g.nu * lap - g.alpha * w + g.forcing


def step(g: FVGrid, w, dt: float, mode: str, held, base: str = "central"):
    """SSP-RK3; the held quantity is constant within the step."""
    w1 = w + dt * rhs(g, w, mode, held, base)
    w2 = 0.75 * w + 0.25 * (w1 + dt * rhs(g, w1, mode, held, base))
    return w / 3.0 + 2.0 / 3.0 * (w2 + dt * rhs(g, w2, mode, held, base))


def base_name(model) -> str:
    return model.get("static", closure.Static()).base if model is not None else "central"


def refresh(g: FVGrid, model, w, mode: str):
    if mode in ("upwind3", "central"):
        return jnp.zeros((), w.dtype)
    coef = li_coefficients(g, model, w)
    if mode == "li":
        return coef
    base = base_name(model)  # li_value: the learned flux correction's tendency, held
    ux, vy = face_velocities(g, w)
    return (advection(g, ux, vy, *interp_li(coef, w, ux, vy, base))
            - advection(g, ux, vy, *base_of(base, w, ux, vy)))


def advect_cells(g: FVGrid, field, w, dt: float):
    """Carry a held cell field (B, n, n, C) with the flow for one step (semi-Lagrangian).

    The held learned operator then stays attached to the structures it was computed
    for, as MD's per-atom residual stays attached to its atom. On the fixed grid a
    held stencil is a cell out of place after 2-3 steps at CFL 0.4.
    """
    ux, vy = face_velocities(g, w)
    uc = 0.5 * (ux + sh(ux, -1, 0))
    vc = 0.5 * (vy + sh(vy, 0, -1))
    ii, jj = jnp.meshgrid(jnp.arange(g.n, dtype=jnp.float32), jnp.arange(g.n, dtype=jnp.float32),
                          indexing="ij")
    si, sj = ii - uc * dt / g.dx, jj - vc * dt / g.dx

    def one(f, a, b):
        return map_coordinates(f, [a, b], order=1, mode="wrap")

    return jax.vmap(jax.vmap(one, in_axes=(-1, None, None), out_axes=-1))(field, si, sj)


ADVECTED = {"li_adv": "li", "li_value_adv": "li_value"}
STUDENT_MODES = ("li_student", "li_res", "li_res_adv")


def _student(model):
    return {"params": model["student"], "scale": model["scale"], "static": model.get("static")}


def outer_step(g: FVGrid, model, w, mode: str, k: int, dt: float):
    """One refresh, then k steps.

    The student modes are Frost's multirate in CFD. A small network (the student)
    gives the learned stencil every step; the large one (the teacher) is called once
    per k steps, and its residual R = c_T - c_S is held (``li_res``) or carried with
    the flow (``li_res_adv``):

        c = c_S(w_now) + [c_T(w_ref) - c_S(w_ref)]

    ``li_student`` is the student alone, every step.
    """
    base = base_name(model)
    if mode in STUDENT_MODES:
        stu = _student(model)
        if mode == "li_student":
            body = lambda x, _: (step(g, x, dt, "li", li_coefficients(g, stu, x), base), None)  # noqa: E731
            return jax.lax.scan(body, w, None, length=k)[0]
        res = li_coefficients(g, model, w) - li_coefficients(g, stu, w)

        def body(carry, _):
            x, r = carry
            nxt = step(g, x, dt, "li", li_coefficients(g, stu, x) + r, base)
            if mode == "li_res_adv":
                r = advect_cells(g, r, x, dt)
            return (nxt, r), None

        return jax.lax.scan(body, (w, res), None, length=k)[0][0]
    if mode in ADVECTED:
        inner = ADVECTED[mode]
        held = refresh(g, model, w, inner)
        if inner == "li_value":
            held = held[..., None]

        def body(carry, _):
            x, hd = carry
            use = hd[..., 0] if inner == "li_value" else hd
            return (step(g, x, dt, inner, use, base), advect_cells(g, hd, x, dt)), None

        return jax.lax.scan(body, (w, held), None, length=k)[0][0]
    held = refresh(g, model, w, mode)
    return jax.lax.scan(lambda x, _: (step(g, x, dt, mode, held, base), None), w, None, length=k)[0]


@partial(jax.jit, static_argnames=("g", "mode", "k", "dt", "n_records", "record_every"))
def rollout(g: FVGrid, model, w, *, mode: str, k: int, dt: float, n_records: int, record_every: int):
    if record_every % k:
        raise ValueError("record_every must be a multiple of k")

    def record(x, _):
        x = jax.lax.scan(lambda y, _: (outer_step(g, model, y, mode, k, dt), None), x, None,
                         length=record_every // k)[0]
        return x, x

    return jax.lax.scan(record, w, None, length=n_records)[1]


@partial(jax.jit, static_argnames=("g", "mode", "k", "dt", "n_outer"))
def advance(g: FVGrid, model, w, *, mode: str, k: int, dt: float, n_outer: int):
    return jax.lax.scan(lambda y, _: (outer_step(g, model, y, mode, k, dt), None), w, None,
                        length=n_outer)[0]


# ------------------------------------------------------------------ certified intervals
def _run_hold(g: FVGrid, coef, w, k: int, dt: float, base: str):
    return jax.lax.scan(lambda x, _: (step(g, x, dt, "li", coef, base), None), w, None, length=k)[0]


def _hold_error(g: FVGrid, c_held, c_fresh, w, base: str):
    """e = ||T(held) - T(fresh)|| / ||RHS||, per batch member: the hold's own error, measured
    at the interval's end with the refresh the next interval needs anyway (no extra CNN call)."""
    ux, vy = face_velocities(g, w)
    t_held = advection(g, ux, vy, *interp_li(c_held, w, ux, vy, base))
    t_fresh = advection(g, ux, vy, *interp_li(c_fresh, w, ux, vy, base))
    full = rhs(g, w, "li", c_fresh, base)
    num = jnp.sqrt(jnp.mean((t_held - t_fresh) ** 2, axis=(-2, -1)))
    return num / jnp.sqrt(jnp.mean(full**2, axis=(-2, -1)))


def _certified_body(g: FVGrid, model, k: int, dt: float, e_star: float):
    """One certified interval: hold k steps; if any member's hold error exceeds e_star (or is
    non-finite), redo the interval from its start as two halves with a refresh between.
    The certificate of plan rule (b), the CFD counterpart of MD's --rollback."""
    base = base_name(model)

    def body(carry, _):
        w, c, n_rb = carry
        wk = _run_hold(g, c, w, k, dt, base)
        ck = li_coefficients(g, model, wk)
        e = _hold_error(g, c, ck, wk, base)
        bad = ~jnp.isfinite(e) | (e > e_star) | ~jnp.all(jnp.isfinite(wk), axis=(-2, -1))

        def redo(_):
            h = k // 2
            wm = _run_hold(g, c, w, h, dt, base)
            wk2 = _run_hold(g, li_coefficients(g, model, wm), wm, k - h, dt, base)
            ck2 = li_coefficients(g, model, wk2)
            sel = bad[:, None, None]
            return jnp.where(sel, wk2, wk), jnp.where(sel[..., None], ck2, ck)

        wk, ck = jax.lax.cond(jnp.any(bad), redo, lambda _: (wk, ck), None)
        return (wk, ck, n_rb + jnp.any(bad).astype(jnp.int32)), jnp.mean(e)

    return body


@partial(jax.jit, static_argnames=("g", "k", "dt", "e_star", "n_records", "record_every"))
def cert_rollout(g: FVGrid, model, w, *, k: int, dt: float, e_star: float, n_records: int,
                 record_every: int):
    body = _certified_body(g, model, k, dt, e_star)

    def record(carry, _):
        carry, es = jax.lax.scan(body, carry, None, length=record_every // k)
        return carry, (carry[0], jnp.mean(es))

    init = (w, li_coefficients(g, model, w), jnp.zeros((), jnp.int32))
    (_, _, n_rb), (states, es) = jax.lax.scan(record, init, None, length=n_records)
    return states, n_rb, jnp.mean(es)


@partial(jax.jit, static_argnames=("g", "k", "dt", "e_star", "n_outer"))
def cert_advance(g: FVGrid, model, w, *, k: int, dt: float, e_star: float, n_outer: int):
    body = _certified_body(g, model, k, dt, e_star)
    init = (w, li_coefficients(g, model, w), jnp.zeros((), jnp.int32))
    (w, _, n_rb), _ = jax.lax.scan(body, init, None, length=n_outer)
    return w
