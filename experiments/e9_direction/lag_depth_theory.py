"""How deep can the teacher pipeline go? Stability of lag-L residual kicks (new_directions.md, R2).

With L teacher devices working round-robin, a new residual still arrives every
outer step, but it was computed L steps ago, so the kick at boundary n uses
R(x_{n-L}). The wall time per outer step falls to max(k c_S, c_T / L): the teacher
term shrinks with L until the student alone bounds the speedup.

For one harmonic mode, x_{n-L} = x_n cos(L theta) - (v_n / w) sin(L theta), so the
lagged kick carries a velocity term mu w D sin(L theta) v_n. That is negative
friction whose amplitude growth is about 1 + mu theta sin(L theta) / 2 per outer
step. A system has modes at every theta up to the fastest, so what matters is the
worst mode:

    growth_L = max over theta in (0, theta_max] of the exact spectral radius,

with theta_max = 2 pi H / T_fastest (0.44 pi at Frost's 0.22 T). This computes the
exact radius of the lag-L map, the friction gamma D needed to stabilise every mode,
and the resulting wall-clock ceiling from measured costs.

Output: data/results/e9/lag_depth_theory.json

Usage
  LANE=check scripts/safe_run.sh e9_lag_depth .venvs/frost-mace/bin/python \\
      experiments/e9_direction/lag_depth_theory.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "results" / "e9" / "lag_depth_theory.json"


def radius(theta: float, mu: float, lag: int, gamma_d: float = 0.0) -> float:
    """Spectral radius of one outer step, state (x_n, v_n, x_{n-1}, ..., x_{n-lag})."""
    w, big = 1.0, theta
    c, s = math.cos(theta), math.sin(theta)
    damp = math.exp(-gamma_d)
    size = 2 + lag
    kick = np.zeros(size)
    kick[1] = 1.0
    kick[1 + lag if lag else 0] += -mu * w * w * big  # the residual source x_{n-lag}
    x_row = np.zeros(size)
    x_row[0] = 1.0
    step = np.zeros((size, size))
    step[0] = c * x_row + (s / w) * kick
    step[1] = (-w * s * damp) * x_row + (c * damp) * kick
    if lag:
        step[2] = x_row  # new x_{n-1} = x_n
        for j in range(3, size):
            step[j, j - 1] = 1.0  # shift the older lags
    return float(np.max(np.abs(np.linalg.eigvals(step))))


def worst(mu: float, lag: int, theta_max: float, gamma_d: float = 0.0) -> tuple[float, float]:
    thetas = np.linspace(0.01, theta_max, 400)
    radii = [radius(t, mu, lag, gamma_d) for t in thetas]
    i = int(np.argmax(radii))
    return radii[i], float(thetas[i])


def friction_needed(mu: float, lag: int, theta_max: float) -> float | None:
    for g in np.linspace(0.0, 1.0, 1001):
        if worst(mu, lag, theta_max, g)[0] <= 1.0 + 1e-10:
            return float(g)
    return None


def main() -> int:
    theta_max = 0.44 * math.pi
    report: dict = {"theta_max_over_pi": 0.44, "rows": []}
    for mu in (0.01, 0.025, 0.05):
        for lag in (0, 1, 2, 3, 4):
            r, at = worst(mu, lag, theta_max)
            g = friction_needed(mu, lag, theta_max) if lag else 0.0
            row = {"mu": mu, "lag": lag, "worst_radius": r, "at_theta_over_pi": at / math.pi,
                   "predicted_1st_order": 1 + 0.5 * mu * max(
                       t * math.sin(lag * t) for t in np.linspace(0.01, theta_max, 400)) if lag else 1.0,
                   "gamma_D_needed": g}
            report["rows"].append(row)
            print(f"mu {mu:5.3f} lag {lag}: worst radius {r:.5f} at theta {at / math.pi:.3f} pi"
                  f" (first order {row['predicted_1st_order']:.5f}) | friction gamma*D needed "
                  f"{g}", flush=True)
    # wall-clock ceiling from measured costs (plan section 1.15 and new_directions.md section 1)
    systems = {"silica": (0.0755, 1 / 6), "li3po4": (0.1036, 1 / 6), "si": (0.218, 1 / 6),
               "cu": (0.109, 1 / 12)}
    report["ceiling"] = {}
    for name, (c_s, dt_over_h) in systems.items():
        report["ceiling"][name] = {f"L={lag}": 1.0 / max(c_s, dt_over_h / max(lag, 1)) if lag
                                   else 1.0 / (c_s + dt_over_h) for lag in (0, 1, 2, 3, 4)}
        print(name, {k: round(v, 2) for k, v in report["ceiling"][name].items()})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
