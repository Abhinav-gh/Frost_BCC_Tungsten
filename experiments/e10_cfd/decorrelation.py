"""How long can a learned output be held? The hold error e(k), measured (CFD half).

The quantity, the same in both domains: along the workload's own every-step (B0)
trajectory, hold the learned quantity computed at step t, use it at step t + k,
and compare with a fresh evaluation there. Relative to the full time derivative:

    e(k) = || T_hold(t; t+k) - T_fresh(t+k) || / || RHS(t+k) ||,

where T is the learned contribution to d omega/dt and RHS the whole of d omega/dt.
The MD half (``experiments/e9_direction/residual_decorrelation.py``) uses the
residual's contribution to the acceleration over the whole acceleration.

Also reported, D(k) = <|y(t+k) - y(t)|^2> / (2 var y): the learned output's own
decorrelation (0 = identical, 1 = unrelated).

Measured here:
- **W1 spectral** (structured tensor closure): operator and value holds, both
  Eulerian and carried with the flow.
- **W1 finite volume** (learned interpolation teacher): operator and value holds,
  Eulerian and carried; and the residual over each distilled student
  (c_S(now) + [c_T - c_S](held)), Eulerian and carried.

The question is whether one threshold on e(k) separates every configuration
measured to pass at matched accuracy from every one measured to fail (tests 12
and the W1-FV runs).

Output: data/results/e10/decorrelation_cfd.json

Usage (guarded launcher; GPU, JAX env)
  scripts/safe_run.sh cfd_decor env -u LD_LIBRARY_PATH -u CUDA_HOME PYTHONPATH=src \\
      .venvs/frost-mhd/bin/python experiments/e10_cfd/decorrelation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from frost.cfd import closure, spectral
from frost.cfd import finite_volume as fv
from frost.cfd.spectral import Grid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_closure import DATA, load_model  # noqa: E402
from w1fv import half_cell  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "results" / "e10" / "decorrelation_cfd.json"
MODELS = ROOT / "data" / "models" / "cfd"
DT = 0.007
LAGS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
STARTS = tuple(range(0, 64, 8))


def rms(x):
    return jnp.sqrt(jnp.mean(x**2))


def selfdecor(y_t, y_tk):
    """D(k) over all cells and channels, against the variance of the whole field."""
    var = jnp.var(jnp.concatenate([y_t.reshape(-1), y_tk.reshape(-1)]))
    return jnp.mean((y_tk - y_t) ** 2) / (2 * var)


def spectral_part(ev0: np.ndarray) -> dict:
    g = Grid(64)
    model = load_model(MODELS / "kolmogorov64_flux_cnn")
    w = jnp.asarray(np.fft.rfft2(ev0) * g.mask, jnp.complex64)

    @jax.jit
    def b0(wh):
        coef = closure.coefficients(g, model, wh)
        return spectral.step(g, wh, DT, lambda x: closure.apply_operator(g, coef, x))

    traj = [w]
    for _ in range(max(STARTS) + max(LAGS)):
        traj.append(b0(traj[-1]))

    coef_of = jax.jit(lambda wh: closure.coefficients(g, model, wh))
    tend = jax.jit(lambda c, wh: spectral.to_real(g, closure.apply_operator(g, c, wh)))

    @jax.jit
    def full_rhs(wh):
        c = closure.coefficients(g, model, wh)
        out = spectral.advection_hat(g, wh) + g.forcing_hat + g.linear * wh + closure.apply_operator(g, c, wh)
        return spectral.to_real(g, out * g.mask)

    adv = jax.jit(lambda f, wh: closure.advect(g, f, wh, DT))
    res = {m: {k: [] for k in LAGS} for m in ("operator", "value", "operator_adv", "value_adv")}
    dk = {k: [] for k in LAGS}
    for t in STARTS:
        c0 = coef_of(traj[t])
        v0 = tend(c0, traj[t])[..., None]
        c_adv, v_adv, step_k = c0, v0, 0
        for k in LAGS:
            while step_k < k:  # carry the held fields along the trajectory one step at a time
                c_adv = adv(c_adv, traj[t + step_k])
                v_adv = adv(v_adv, traj[t + step_k])
                step_k += 1
            wk = traj[t + k]
            ck = coef_of(wk)
            fresh = tend(ck, wk)
            denom = rms(full_rhs(wk))
            res["operator"][k].append(float(rms(tend(c0, wk) - fresh) / denom))
            res["value"][k].append(float(rms(v0[..., 0] - fresh) / denom))
            res["operator_adv"][k].append(float(rms(tend(c_adv, wk) - fresh) / denom))
            res["value_adv"][k].append(float(rms(v_adv[..., 0] - fresh) / denom))
            dk[k].append(float(selfdecor(c0, ck)))
    share = float(rms(tend(coef_of(traj[0]), traj[0])) / rms(full_rhs(traj[0])))
    return {"e": {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in res.items()},
            "D_output": {k: float(np.mean(v)) for k, v in dk.items()},
            "learned_share_of_rhs": share}


def fv_part(ev0_cell: np.ndarray) -> dict:
    g = fv.FVGrid(64)
    teacher = load_model(MODELS / "kolmogorov64_fv_li")
    meta = json.loads((MODELS / "kolmogorov64_fv_li.json").read_text())
    static = closure.Static(base=meta.get("base", "central"))
    teacher["static"] = static
    base = static.base
    students = {}
    for name in ("s3x16", "s4x32"):
        path = MODELS / f"kolmogorov64_fv_li_{name}"
        if path.with_suffix(".npz").exists():
            s = load_model(path)
            s["static"] = static
            students[name] = s

    w = jnp.asarray(ev0_cell)

    @jax.jit
    def b0(x):
        return fv.step(g, x, DT, "li", fv.li_coefficients(g, teacher, x), base)

    traj = [w]
    for _ in range(max(STARTS) + max(LAGS)):
        traj.append(b0(traj[-1]))

    coef_t = jax.jit(lambda x: fv.li_coefficients(g, teacher, x))
    coef_s = {n: jax.jit(lambda x, m=m: fv.li_coefficients(g, m, x)) for n, m in students.items()}

    @jax.jit
    def learned(c, x):
        """The learned flux correction's tendency: the held stencil applied to state x."""
        ux, vy = fv.face_velocities(g, x)
        return (fv.advection(g, ux, vy, *fv.interp_li(c, x, ux, vy, base))
                - fv.advection(g, ux, vy, *fv.base_of(base, x, ux, vy)))

    full_rhs = jax.jit(lambda x: fv.rhs(g, x, "li", fv.li_coefficients(g, teacher, x), base))
    adv = jax.jit(lambda f, x: fv.advect_cells(g, f, x, DT))
    modes = ["operator", "value", "operator_adv", "value_adv"]
    modes += [f"residual_{n}{suf}" for n in students for suf in ("", "_adv")]
    res = {m: {k: [] for k in LAGS} for m in modes}
    dk = {k: [] for k in LAGS}
    dk_res = {n: {k: [] for k in LAGS} for n in students}
    for t in STARTS:
        x0 = traj[t]
        c0 = coef_t(x0)
        v0 = learned(c0, x0)[..., None]
        r0 = {n: c0 - coef_s[n](x0) for n in students}
        carried = {"c": c0, "v": v0, **{f"r_{n}": r0[n] for n in students}}
        step_k = 0
        for k in LAGS:
            while step_k < k:
                carried = {key: adv(val, traj[t + step_k]) for key, val in carried.items()}
                step_k += 1
            xk = traj[t + k]
            ck = coef_t(xk)
            fresh = learned(ck, xk)
            denom = rms(full_rhs(xk))
            res["operator"][k].append(float(rms(learned(c0, xk) - fresh) / denom))
            res["value"][k].append(float(rms(v0[..., 0] - fresh) / denom))
            res["operator_adv"][k].append(float(rms(learned(carried["c"], xk) - fresh) / denom))
            res["value_adv"][k].append(float(rms(carried["v"][..., 0] - fresh) / denom))
            dk[k].append(float(selfdecor(c0, ck)))
            for n in students:
                csk = coef_s[n](xk)
                res[f"residual_{n}"][k].append(float(rms(learned(csk + r0[n], xk) - fresh) / denom))
                res[f"residual_{n}_adv"][k].append(
                    float(rms(learned(csk + carried[f"r_{n}"], xk) - fresh) / denom))
                dk_res[n][k].append(float(selfdecor(r0[n], ck - csk)))
    share = float(rms(learned(coef_t(traj[0]), traj[0])) / rms(full_rhs(traj[0])))
    return {"e": {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in res.items()},
            "D_output": {k: float(np.mean(v)) for k, v in dk.items()},
            "D_residual": {n: {k: float(np.mean(v)) for k, v in d.items()} for n, d in dk_res.items()},
            "learned_share_of_rhs": share}


def main() -> int:
    with h5py.File(DATA, "r") as fh:
        ev0 = np.asarray(fh["eval/w64"][0])
    print(f"devices {jax.devices()} | lags {LAGS} (steps of {DT}) | starts {STARTS}", flush=True)
    out = {"lags_steps": LAGS, "dt": DT}
    for name, fn, arg in (("w1_spectral", spectral_part, ev0), ("w1_fv", fv_part, half_cell(ev0))):
        r = fn(arg)
        out[name] = r
        print(f"== {name}: learned share of the RHS {r['learned_share_of_rhs']:.3f}", flush=True)
        for m, d in r["e"].items():
            print(f"  e(k) {m:18s} " + " ".join(f"{k}:{d[k]:.3f}" for k in LAGS), flush=True)
        print(f"  D(k) output          " + " ".join(f"{k}:{r['D_output'][k]:.3f}" for k in LAGS), flush=True)
        for n, d in r.get("D_residual", {}).items():
            print(f"  D(k) residual {n:6s} " + " ".join(f"{k}:{d[k]:.3f}" for k in LAGS), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
