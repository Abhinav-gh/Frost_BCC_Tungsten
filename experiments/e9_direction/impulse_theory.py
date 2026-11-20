"""Numerical checks of the multirate results in docs/design/research_direction.md (section D).

Four claims about impulse r-RESPA with a student inner force and a teacher residual
R = F_T - F_S, each checked here against its closed form:

1. Linear resonance bands. For a harmonic mode with student frequency w and a
   residual stiffness mu * w^2 (mu = (w_T^2 - w_S^2) / w_S^2), one outer step of
   length D is M = K Phi K with trace 2 cos(t) - mu t sin(t), t = w D. It is
   unstable in a band of width |mu| m pi next to every t = m pi, with growth
   1 + |mu| m pi / 2 per outer step. A better student (smaller |mu|) narrows every
   band and never removes one.

2. The free endpoint estimator is blind where impulse error is largest. For a
   residual component a cos(w t), the endpoint difference the controller measures
   has amplitude 2 a |sin(t/2)|, while the impulse (trapezoid) momentum error per
   outer step has amplitude a D |sinc(t/2) - cos(t/2)|. Their ratio diverges at
   t = 2 pi m: the estimator reads zero while the whole impulse is spurious.

3. Nonlinear resonances survive a perfect curvature match. With a Morse teacher and
   a harmonic student of the same curvature (mu = 0, so claim 1 predicts no band at
   all), impulse RESPA still blows up in windows near D = T/n for n = 3, 4, 5, the
   steps where Ma, Izaguirre and Skeel (2003) locate impulse MTS's nonlinear
   instabilities, while the endpoint estimator reads the same on unstable steps as
   on the stable steps between them.

4. Differential corrections. Extrapolating the residual linearly in time from the
   last two outer samples beats holding it, at the end of the interval, only when
   t < pi / 3 (D < T/6 of the mode that carries the staleness); endpoint-derivative
   corrections of any order vanish on modes with t = 2 pi m, so none can see the
   aliased modes of claim 2.

Everything is numpy on a scalar or 1-D model: no model is evaluated.

Output: data/results/e9/impulse_theory.json

Usage
  LANE=check scripts/safe_run.sh e9_impulse_theory .venvs/frost-mace/bin/python \\
      experiments/e9_direction/impulse_theory.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "results" / "e9" / "impulse_theory.json"


def bands(mu: float, t_max: float = 4.5 * math.pi, n: int = 2_000_001) -> list[dict]:
    """Unstable intervals of 2 cos t - mu t sin t, found on a fine grid."""
    t = np.linspace(1e-6, t_max, n)
    tr = 2 * np.cos(t) - mu * t * np.sin(t)
    unstable = np.abs(tr) > 2.0
    edges = np.flatnonzero(np.diff(unstable.astype(int)))
    out = []
    for lo, hi in zip(edges[::2], edges[1::2], strict=False):
        seg = slice(lo + 1, hi + 1)
        m = round(t[seg].mean() / math.pi)
        a = np.abs(tr[seg]).max()
        growth = (a + math.sqrt(a * a - 4.0)) / 2.0
        out.append({
            "m": m,
            "lo_over_pi": float(t[lo + 1] / math.pi),
            "hi_over_pi": float(t[hi] / math.pi),
            "width_over_pi": float((t[hi] - t[lo + 1]) / math.pi),
            "predicted_width_over_pi": abs(mu) * m,
            "growth_per_outer_step": float(growth),
            "predicted_growth": 1.0 + abs(mu) * m * math.pi / 2.0,
        })
    return out


def estimator_ratio(theta: np.ndarray) -> np.ndarray:
    """Impulse error per outer step over (D x endpoint estimator), phase-averaged, one mode."""
    q = np.abs(np.sinc(theta / (2 * math.pi)) - np.cos(theta / 2))  # np.sinc is sin(pi x)/(pi x)
    e = 2 * np.abs(np.sin(theta / 2))
    return q / np.maximum(e, 1e-300)


def morse_scan(fractions: np.ndarray, energy: float = 0.08, periods: int = 400,
               inner_per_period: int = 400) -> list[dict]:
    """Teacher V_T = (1 - exp(-x))^2 (D = alpha = m = 1), student V_S = x^2 (same curvature).

    The residual R = F_T - F_S is purely anharmonic. Along the teacher's own orbit at
    the given energy, compare for each outer step D:
      - the RMS endpoint difference |R(t+D) - R(t)| (what the controller measures);
      - the RMS trapezoid error |int R dt - D/2 (R(t) + R(t+D))| (what the impulse does);
    and run impulse RESPA itself (exact harmonic inner flow) to record its energy error.
    """

    def f_t(x):
        ex = np.exp(-x)
        return -2.0 * (1.0 - ex) * ex

    def f_s(x):
        return -2.0 * x

    def residual(x):
        return f_t(x) - f_s(x)

    w = math.sqrt(2.0)
    t_s = 2 * math.pi / w
    # teacher orbit by fine velocity Verlet, from the turning point at the given energy
    x_turn = -math.log(1.0 + math.sqrt(energy))
    dt = t_s / 4000.0
    n = int(periods * 4000)
    x = np.empty(n)
    xi, vi = x_turn, 0.0
    fi = f_t(xi)
    for k in range(n):
        x[k] = xi
        vi += 0.5 * dt * fi
        xi += dt * vi
        fi = f_t(xi)
        vi += 0.5 * dt * fi
    r = residual(x)
    cum = np.concatenate(([0.0], np.cumsum(0.5 * dt * (r[1:] + r[:-1]))))
    period_t = teacher_period(x, dt)

    rows = []
    for frac in fractions:
        lag = max(1, round(frac * period_t / dt))
        big = lag * dt
        endpoint = r[lag:] - r[:-lag]
        integral = cum[lag:] - cum[:-lag]
        trap = 0.5 * big * (r[lag:] + r[:-lag])
        quad = integral - trap
        rows.append({
            "D_over_T": float(big / period_t),
            "endpoint_rms": float(np.sqrt(np.mean(endpoint**2))),
            "impulse_error_rms": float(np.sqrt(np.mean(quad**2))),
            "ratio": float(np.sqrt(np.mean(quad**2)) / (big * np.sqrt(np.mean(endpoint**2)))),
            "respa_energy_error": respa_energy_error(big, w, f_t, f_s, x_turn, periods,
                                                     inner_per_period, period_t),
        })
    return rows


def teacher_period(x: np.ndarray, dt: float) -> float:
    up = np.flatnonzero((x[:-1] < 0) & (x[1:] >= 0))
    return float(np.mean(np.diff(up)) * dt)


def respa_energy_error(big, w, f_t, f_s, x0, periods, inner_per_period, period_t):
    """Max |H_T - H_T(0)| over the run for impulse RESPA with exact harmonic inner flow."""
    del inner_per_period  # the harmonic student is integrated exactly
    c, s = math.cos(w * big), math.sin(w * big)
    x, v = x0, 0.0

    def h_t(x, v):
        return 0.5 * v * v + (1.0 - math.exp(-x)) ** 2

    h0 = h_t(x, v)
    worst = 0.0
    steps = int(periods * period_t / big)
    for _ in range(steps):
        v += 0.5 * big * (f_t(x) - f_s(x))
        x, v = c * x + s / w * v, -w * s * x + c * v
        v += 0.5 * big * (f_t(x) - f_s(x))
        err = abs(h_t(x, v) - h0)
        worst = max(worst, err)
        if not math.isfinite(err) or err > 10.0:
            return float("inf")
    return worst


def extrapolation_vs_hold(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-mode MSE of linear-in-time extrapolation over holding: at the end, and averaged."""
    end = 4 * np.sin(theta / 2) ** 2  # |e^{it} - 2 + e^{-it}|^2 / |e^{it} - 1|^2
    tau = np.linspace(0.0, 1.0, 401)[None, :]
    th = theta[:, None]
    u = np.exp(1j * th * tau) - 1.0
    wv = 1.0 - np.exp(-1j * th)
    lin = np.abs(u - tau * wv) ** 2
    hold = np.abs(u) ** 2
    avg = np.trapezoid(lin, tau, axis=1) / np.maximum(np.trapezoid(hold, tau, axis=1), 1e-300)
    return end, avg


def first_crossing(theta: np.ndarray, values: np.ndarray, level: float) -> float:
    idx = np.flatnonzero(values > level)
    return float(theta[idx[0]]) if idx.size else float("nan")


def main() -> int:
    report: dict = {"bands": {}}
    for mu in (0.02, 0.1, 0.3, -0.1):
        report["bands"][str(mu)] = bands(mu)

    theta = np.linspace(1e-3, 4.2 * math.pi, 20_001)
    ratio = estimator_ratio(theta)
    report["estimator"] = {
        "ratio_small_theta_over_theta": float(ratio[10] / theta[10]),  # expect 1/12
        "theta_where_ratio_exceeds_1_over_pi": first_crossing(theta, ratio, 1.0) / math.pi,
        "ratio_at": {f"{f:.2f}pi": float(estimator_ratio(np.array([f * math.pi]))[0])
                     for f in (0.25, 0.44, 0.5, 1.0, 1.5, 1.9, 1.99, 2.0)},
    }

    end, avg = extrapolation_vs_hold(theta)
    report["extrapolation"] = {
        "end_crossing_over_pi": first_crossing(theta, end, 1.0) / math.pi,  # expect 1/3
        "interval_average_crossing_over_pi": first_crossing(theta, avg, 1.0) / math.pi,
    }
    # the E7 table's cases, as theta = 2 pi D / T for the fastest period of each system
    cases = {"silica_k2": (2, 27.7), "silica_k4": (4, 27.7), "li3po4_k2": (2, 33.0),
             "li3po4_k4": (4, 33.0), "water_k2_OH": (2, 9.0)}
    report["extrapolation"]["e7_cases"] = {}
    for name, (d, period) in cases.items():
        th = np.array([2 * math.pi * d / period])
        e_end, e_avg = extrapolation_vs_hold(th)
        report["extrapolation"]["e7_cases"][name] = {
            "theta_over_pi": float(th[0] / math.pi),
            "end_ratio": float(e_end[0]),
            "average_ratio": float(e_avg[0]),
        }

    fractions = np.round(np.arange(0.08, 0.5601, 0.004), 4)
    report["morse"] = {}
    for energy in (0.02, 0.05, 0.08):
        rows = morse_scan(fractions, energy=energy)
        unstable = [r for r in rows if not r["respa_energy_error"] < energy]
        stable = [r for r in rows if r["respa_energy_error"] < energy]
        report["morse"][str(energy)] = {
            "rows": rows,
            "unstable_D_over_T": [round(r["D_over_T"], 3) for r in unstable],
            "endpoint_rms_stable": [min(r["endpoint_rms"] for r in stable),
                                    max(r["endpoint_rms"] for r in stable)] if stable else None,
            "endpoint_rms_unstable": [min(r["endpoint_rms"] for r in unstable),
                                      max(r["endpoint_rms"] for r in unstable)] if unstable else None,
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))

    for mu, rows in report["bands"].items():
        print(f"mu = {mu}")
        for b in rows:
            print(f"  m={b['m']}  [{b['lo_over_pi']:.4f}, {b['hi_over_pi']:.4f}] pi  width "
                  f"{b['width_over_pi']:.4f} (pred {b['predicted_width_over_pi']:.4f})  growth "
                  f"{b['growth_per_outer_step']:.4f} (pred {b['predicted_growth']:.4f})")
    print("estimator", json.dumps(report["estimator"], indent=1))
    print("extrapolation", json.dumps(report["extrapolation"], indent=1))
    for energy, block in report["morse"].items():
        print(f"Morse teacher, harmonic student, E = {energy}:")
        print("  unstable D/T:", block["unstable_D_over_T"])
        print("  endpoint RMS on stable steps", block["endpoint_rms_stable"],
              "on unstable steps", block["endpoint_rms_unstable"])
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
