"""CFD tests of the reframed runtime (new_directions.md R11, tests 3-5), on astronomix.

Three questions about a learned component inside an MHD solver, each asked on two
inhomogeneous 2D problems:
- the Orszag-Tang vortex (current sheets);
- the Gardiner-Stone MHD blast wave (a front).

Test 3 (R8, spatial rate clusters). Is the refresh demand concentrated? Per solver
step, the change of the closure's inputs (the velocity and magnetic gradient
tensors) is aggregated over 16x16-cell blocks. Reported: the fraction of blocks
carrying 50% and 80% of the change, at intervals of 1 and 4 steps. MHD_64's
homogeneous turbulence gave 38-42% for 50% (section 13); spatial clusters are
worth building only if these problems are far more concentrated.

Test 4 (R4, the spectral roofline in CFD). How much can reuse of a closure's
*output* gain? The output of a gradient-model closure (Clark: tau_ij = C D^2
du_i/dx_k du_j/dx_k, a stand-in for a learned subgrid stress, clearly a proxy) is
recorded in every cell at every step. The aliased-Wiener bound then gives the
efficiency of holding it for k steps, and of trapezoid use of the k-spaced samples,
against the best linear reuse (Matthews 2000), at k = 2, 4, 8.

Test 5 (R2 in CFD). Does a lagged or held closure destabilise the solver? A
Smagorinsky eddy viscosity (a stand-in for a learned, value-consumed closure; it
acts on momentum only, so div B is untouched) is applied by operator splitting
after each solver step. It is evaluated:
- fresh every step (the reference);
- held for k = 2, 4, 8 steps;
- lagged by L = 1, 2, 4 steps.

Measured: the relative L2 error of (rho, v, B) against the reference, and the
kinetic- and magnetic-energy differences.

The solver is driven only through astronomix's public ``time_integration`` (one
call per step, fixed dt); the closure is our own JAX code. Nothing in astronomix
is modified.

Output: data/results/e9/cfd_multirate.json

Usage (CPU JAX; small 2D grids)
  LANE=check scripts/safe_run.sh e9_cfd env -u LD_LIBRARY_PATH -u CUDA_HOME \\
      JAX_PLATFORMS=cpu .venvs/frost-mhd/bin/python experiments/e9_direction/cfd_multirate.py
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "results" / "e9" / "cfd_multirate.json"
BLOCK = 16


# ------------------------------------------------------------------ problems
def orszag_tang(n: int):
    x = (np.arange(n) + 0.5) / n
    xx, yy = np.meshgrid(x, x, indexing="ij")
    b0 = 1.0 / math.sqrt(4.0 * math.pi)
    fields = {
        "density": np.full((n, n), 25.0 / (36.0 * math.pi)),
        "velocity_x": -np.sin(2 * math.pi * yy),
        "velocity_y": np.sin(2 * math.pi * xx),
        "gas_pressure": np.full((n, n), 5.0 / (12.0 * math.pi)),
        "magnetic_field_x": -b0 * np.sin(2 * math.pi * yy),
        "magnetic_field_y": b0 * np.sin(4 * math.pi * xx),
        "magnetic_field_z": np.zeros((n, n)),
    }
    return fields, 0.5  # t_end


def blast_wave(n: int):
    x = (np.arange(n) + 0.5) / n - 0.5
    xx, yy = np.meshgrid(x, x, indexing="ij")
    r = np.sqrt(xx**2 + yy**2)
    b = 1.0 / math.sqrt(2.0)
    fields = {
        "density": np.ones((n, n)),
        "velocity_x": np.zeros((n, n)),
        "velocity_y": np.zeros((n, n)),
        "gas_pressure": np.where(r < 0.1, 10.0, 0.1),
        "magnetic_field_x": np.full((n, n), b),
        "magnetic_field_y": np.full((n, n), b),
        "magnetic_field_z": np.zeros((n, n)),
    }
    return fields, 0.15


PROBLEMS = {"orszag_tang": orszag_tang, "blast_wave": blast_wave}


# ------------------------------------------------------------------ solver adapter
class Solver:
    """One fixed-dt solver step through astronomix's public time_integration."""

    def __init__(self, fields: dict, n: int, dt: float):
        import jax.numpy as jnp
        from astronomix.initial_condition_generation.construct_primitive_state import (
            construct_primitive_state,
        )
        from astronomix.option_classes.simulation_config import (
            FINITE_VOLUME,
            HLL,
            PERIODIC_BOUNDARY,
            BoundarySettings,
            BoundarySettings1D,
            SimulationConfig,
            finalize_config,
        )
        from astronomix.option_classes.simulation_params import SimulationParams
        from astronomix.time_stepping import time_integration
        from astronomix.variable_registry.registered_variables import (
            get_registered_variables,
        )

        self.jnp = jnp
        self._time_integration = time_integration
        periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
        config = SimulationConfig(
            solver_mode=FINITE_VOLUME, dimensionality=2, mhd=True, box_size=1.0,
            num_cells=n, riemann_solver=HLL,
            boundary_settings=BoundarySettings(x=periodic, y=periodic),
            fixed_timestep=True, num_timesteps=1,
        )
        self.rv = get_registered_variables(config)
        state = construct_primitive_state(
            config=config, registered_variables=self.rv,
            **{k: jnp.asarray(v, dtype=jnp.float32) for k, v in fields.items()})
        self.config = finalize_config(config, state.shape)
        self.params = SimulationParams(t_end=dt, C_cfl=0.4, gamma=5.0 / 3.0)
        self.state0 = state
        self.dt = dt
        self.dx = 1.0 / n

    def step(self, state):
        return self._time_integration(state, self.config, self.params, self.rv)

    # indices of the 2D FV MHD state: rho, vx, vy, p, Bx, By, Bz
    def fields(self, state):
        return state[0], state[1:3], state[4:7]


# ------------------------------------------------------------------ closures (our code)
def gradients(f, dx, jnp):
    """Periodic central differences of each component: (C, N, N) -> (C, 2, N, N)."""
    gx = (jnp.roll(f, -1, axis=-2) - jnp.roll(f, 1, axis=-2)) / (2 * dx)
    gy = (jnp.roll(f, -1, axis=-1) - jnp.roll(f, 1, axis=-1)) / (2 * dx)
    return jnp.stack([gx, gy], axis=1)


def gradient_model_stress(v, dx, jnp):
    """Clark's gradient-model stress, tau_ij = dx^2 / 12 * du_i/dx_k du_j/dx_k: (2,2,N,N)."""
    g = gradients(v, dx, jnp)
    return (dx * dx / 12.0) * jnp.einsum("ikxy,jkxy->ijxy", g, g)


def smagorinsky_velocity_update(v, dx, dt, jnp, c_s: float = 0.17):
    """Explicit eddy-viscosity increment for velocity: dt * div(nu_t grad v), nu_t = (c_s dx)^2 |S|."""
    g = gradients(v, dx, jnp)
    s = 0.5 * (g + jnp.swapaxes(g, 0, 1))
    mag = jnp.sqrt(2.0 * jnp.sum(s * s, axis=(0, 1)))
    nu = (c_s * dx) ** 2 * mag
    flux = nu[None, None] * g  # (2 comps, 2 dirs, N, N)
    div = ((jnp.roll(flux[:, 0], -1, axis=-2) - jnp.roll(flux[:, 0], 1, axis=-2))
           + (jnp.roll(flux[:, 1], -1, axis=-1) - jnp.roll(flux[:, 1], 1, axis=-1))) / (2 * dx)
    return dt * div


# ------------------------------------------------------------------ tests
def block_change(a, b):
    d = np.asarray(a - b, dtype=np.float64)
    sq = (d.reshape(-1, d.shape[-2], d.shape[-1]) ** 2).sum(0)
    n = sq.shape[0] // BLOCK
    return np.sqrt(sq.reshape(n, BLOCK, n, BLOCK).sum((1, 3)))


def concentration(change, share):
    v = np.sort(change.ravel())[::-1]
    cum = np.cumsum(v) / v.sum()
    return float((np.searchsorted(cum, share) + 1) / v.size)


def reuse_efficiency(series: np.ndarray, k: int) -> dict:
    """Aliased-Wiener efficiency of hold and trapezoid use of k-spaced samples (per-cell mean PSD)."""
    s = np.mean(np.abs(np.fft.fft(series - series.mean(0), axis=0)) ** 2, axis=1)
    n = s.size
    f = np.fft.fftfreq(n)  # cycles per step
    order = np.argsort(f)
    f, s = f[order], s[order]
    # hold (interval-averaged, offsets 1..k) and trapezoid impulse MSE, as in residual_sampling.py
    w = 2 * np.pi * f
    held = np.mean([np.sum(s * np.abs(np.exp(1j * w * j) - 1.0) ** 2) for j in range(1, k + 1)])
    c = np.ones(k + 1)
    c[0] = c[-1] = 0.5
    h = sum(c[j] * np.exp(1j * w * j) for j in range(k + 1))
    trap = np.sum(s * np.abs(h - 0.5 * k * (1 + np.exp(1j * w * k))) ** 2)
    # fold onto one alias band (requires n divisible by k)
    m = n - n % k
    fs, ss, hs = f[:m], s[:m], h[:m]
    band = ss.reshape(k, -1)
    hb = hs.reshape(k, -1)
    fb = fs.reshape(k, -1)
    total = band.sum(0)
    lmmse_imp = np.sum((np.abs(hb) ** 2 * band).sum(0)
                       - np.abs((hb * band).sum(0)) ** 2 / np.maximum(total, 1e-300))
    lmmse_pt = np.mean([np.sum(total - np.abs((band * np.exp(2j * np.pi * fb * j)).sum(0)) ** 2
                               / np.maximum(total, 1e-300)) for j in range(1, k + 1)])
    return {"held_rel_rms": float(math.sqrt(held / max(np.sum(s), 1e-300))),
            "lmmse_over_held": float(math.sqrt(lmmse_pt / held)) if held > 0 else float("nan"),
            "trapezoid_efficiency": float(lmmse_imp / trap) if trap > 0 else float("nan")}


def run_problem(name: str, n: int, window: int, cfl: float) -> dict:
    import jax.numpy as jnp

    fields, t_end = PROBLEMS[name](n)
    # a fixed dt from the initial fast speed, with margin for the flow that develops
    rho, p = fields["density"], fields["gas_pressure"]
    b2 = fields["magnetic_field_x"] ** 2 + fields["magnetic_field_y"] ** 2
    cf = np.sqrt((5.0 / 3.0) * p / rho + b2 / rho).max()
    vmax = np.sqrt(fields["velocity_x"] ** 2 + fields["velocity_y"] ** 2).max()
    dt = cfl * (1.0 / n) / (cf + vmax + 1.0)
    steps = int(t_end / dt)
    solver = Solver(fields, n, dt)
    dx = solver.dx
    start = max(steps - window, 0)
    print(f"{name}: n {n}, dt {dt:.2e}, {steps} steps, window from step {start}", flush=True)

    # reference run with the fresh closure every step; record test-3/4 data in the window
    t0 = time.perf_counter()
    state = solver.state0
    inputs, stress = [], []
    closure_every = []
    for i in range(steps):
        state = solver.step(state)
        _, v, _ = solver.fields(state)
        dv = smagorinsky_velocity_update(v, dx, dt, jnp)
        state = state.at[1:3].add(dv)
        if i >= start:
            _, v, b = solver.fields(state)
            inputs.append((np.asarray(gradients(v, dx, jnp)), np.asarray(gradients(b, dx, jnp))))
            stress.append(np.asarray(gradient_model_stress(v, dx, jnp)).reshape(4, -1))
            closure_every.append(np.asarray(dv))
    ref_final = state
    wall = time.perf_counter() - t0
    if not bool(jnp.all(jnp.isfinite(ref_final))):
        return {"problem": name, "error": "reference run is not finite"}

    # test 3: concentration of input change per block, at intervals 1 and 4 steps
    t3 = {}
    for gap in (1, 4):
        c50, c80 = {"u": [], "b": []}, {"u": [], "b": []}
        for t in range(0, len(inputs) - gap, max(gap, 4)):
            for idx, key in ((0, "u"), (1, "b")):
                ch = block_change(inputs[t + gap][idx], inputs[t][idx])
                c50[key].append(concentration(ch, 0.5))
                c80[key].append(concentration(ch, 0.8))
        t3[f"gap{gap}"] = {k: {"blocks_for_50pct": float(np.mean(c50[k])),
                               "blocks_for_80pct": float(np.mean(c80[k]))} for k in ("u", "b")}

    # test 4: reuse efficiency of the (proxy) closure output
    series = np.stack(stress, axis=0).reshape(len(stress), -1)  # (steps, cells*4)
    usable = len(series) - len(series) % 8
    t4 = {f"k{k}": reuse_efficiency(series[:usable], k) for k in (2, 4, 8)}

    # test 5: held and lagged closure against the fresh reference
    def run_variant(kind: str, value: int):
        s = solver.state0
        history: list = []
        held = None
        for i in range(steps):
            s = solver.step(s)
            _, v, _ = solver.fields(s)
            fresh = smagorinsky_velocity_update(v, dx, dt, jnp)
            if kind == "hold":
                if i % value == 0 or held is None:
                    held = fresh
                s = s.at[1:3].add(held)
            else:  # lag: the closure computed `value` steps ago
                history.append(fresh)
                use = history[-1 - value] if len(history) > value else history[0]
                if len(history) > value + 1:
                    history.pop(0)
                s = s.at[1:3].add(use)
        return s

    def compare(s):
        if not bool(jnp.all(jnp.isfinite(s))):
            return {"finite": False}
        rho_a, v_a, b_a = solver.fields(ref_final)
        rho_b, v_b, b_b = solver.fields(s)

        def rel(a, b):
            return float(jnp.sqrt(jnp.sum((a - b) ** 2) / jnp.sum(a**2)))

        ek = lambda rho, v: float(0.5 * jnp.sum(rho * jnp.sum(v * v, 0)))  # noqa: E731
        em = lambda b: float(0.5 * jnp.sum(b * b))  # noqa: E731
        return {"finite": True, "rel_err_rho": rel(rho_a, rho_b), "rel_err_v": rel(v_a, v_b),
                "rel_err_B": rel(b_a, b_b),
                "kinetic_energy_rel_diff": ek(rho_b, v_b) / ek(rho_a, v_a) - 1.0,
                "magnetic_energy_rel_diff": em(b_b) / em(b_a) - 1.0}

    t5 = {}
    for kind, values in (("hold", (2, 4, 8)), ("lag", (1, 2, 4))):
        for value in values:
            t5[f"{kind}{value}"] = compare(run_variant(kind, value))
            print(f"  {name} {kind}{value}: {t5[f'{kind}{value}']}", flush=True)
    # scale: how large is the closure's effect at all? (no closure against fresh closure)
    s = solver.state0
    for _ in range(steps):
        s = solver.step(s)
    t5["no_closure"] = compare(s)
    print(f"  {name} no_closure: {t5['no_closure']}", flush=True)

    return {"problem": name, "n": n, "dt": dt, "steps": steps, "window": window,
            "reference_wall_s": wall, "test3_concentration": t3, "test4_reuse": t4,
            "test5_hold_lag": t5}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--cfl", type=float, default=0.3)
    ap.add_argument("--problems", nargs="+", default=list(PROBLEMS))
    args = ap.parse_args()
    report = [run_problem(p, args.n, args.window, args.cfl) for p in args.problems]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tag = "" if args.n == 256 else f"_n{args.n}"
    path = OUT.with_name(f"cfd_multirate{tag}.json")
    path.write_text(json.dumps(report, indent=1))
    for r in report:
        if "error" in r:
            print(r)
            continue
        c = r["test3_concentration"]
        print(f"{r['problem']}: test 3 blocks for 50% of change: grad-u {c['gap1']['u']['blocks_for_50pct']:.2f}"
              f" (1 step), {c['gap4']['u']['blocks_for_50pct']:.2f} (4 steps); grad-B "
              f"{c['gap1']['b']['blocks_for_50pct']:.2f} / {c['gap4']['b']['blocks_for_50pct']:.2f}")
        for k, v in r["test4_reuse"].items():
            print(f"  test 4 {k}: held rel RMS {v['held_rel_rms']:.3f}, LMMSE/held "
                  f"{v['lmmse_over_held']:.3f}, trapezoid efficiency {v['trapezoid_efficiency']:.3f}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
