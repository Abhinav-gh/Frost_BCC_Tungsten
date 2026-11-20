"""Can the teacher run one outer step behind, on the second GPU? Stability of lagged residual kicks.

Synchronous Frost pays k student steps plus one teacher call per outer step. With
the teacher on the second A6000, evaluating x_n while the student integrates the
next interval, the wall time per outer step becomes max(k c_S, c_T). The price
is that the kick at boundary n can only use residuals from calls that have
finished: R(x_{n-1}) and older. A kick computed from a past state is a delayed
feedback, and delayed feedback on an oscillator is the textbook route to
instability. Control engineering handles it with the Smith predictor (Smith 1957),
which the cerebellum is thought to implement (Miall et al. 1993); quantum
chemistry faced the same problem when reusing the previous SCF density, and
solved it with a time-reversible extrapolation (Niklasson, Tymczak, Challacombe
2006).

Model: one mode with exact student flow at frequency w, residual force -mu w^2 x,
outer step D (t = w D), optional Langevin friction on the velocity. Each scheme is a
linear map on an augmented state; its spectral radius r per outer step decides
stability (r <= 1). Schemes, all using only finished teacher calls:
  sync        Frost now: kick with R(x_n)            (needs the call at x_n first)
  lag         kick with R(x_{n-1})
  lag_extrap  kick with 2 R(x_{n-1}) - R(x_{n-2})     (linear extrapolation)
  lag_reversible  kick with R(x_{n-1}) + [R(x_{n-1}) - R(x_{n-2})] / 2 ... tested
              as the symmetric Adams-type predictor (3 R_{n-1} - R_{n-2}) / 2

Output: data/results/e9/async_theory.json

Usage
  LANE=check scripts/safe_run.sh e9_async_theory .venvs/frost-mace/bin/python \\
      experiments/e9_direction/async_theory.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "results" / "e9" / "async_theory.json"


def flow(theta: float, w: float, damp: float) -> np.ndarray:
    """Exact harmonic flow over one outer step, then velocity damping exp(-gamma D)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, s / w], [-w * s * damp, c * damp]])


def spectral_radius(scheme: str, theta: float, mu: float, gamma_d: float = 0.0) -> float:
    """State (x_n, v_n, x_{n-1}, x_{n-2}); one outer step = kick then flow, D = theta / w."""
    w = 1.0
    big = theta / w
    damp = math.exp(-gamma_d)
    kick = -mu * w * w * big  # velocity change per unit x of the residual source
    coeffs = {  # weights on (x_n, x_{n-1}, x_{n-2}) in the kick
        "sync": (1.0, 0.0, 0.0),
        "lag": (0.0, 1.0, 0.0),
        "lag_extrap": (0.0, 2.0, -1.0),
        "lag_adams": (0.0, 1.5, -0.5),
    }[scheme]
    # pre-step state (x, v, x1, x2) = (x_n, v_n, x_{n-1}, x_{n-2})
    # kick:  v' = v + kick * (c0 x + c1 x1 + c2 x2)
    # flow:  (x_{n+1}, v_{n+1}) = Phi (x, v')
    # lags:  new x1 = x_n, new x2 = x_{n-1}  (pre-step values)
    kick_row = np.array([kick * coeffs[0], 1.0, kick * coeffs[1], kick * coeffs[2]])
    phi = flow(theta, w, damp)
    step = np.zeros((4, 4))
    x_row = np.array([1.0, 0.0, 0.0, 0.0])
    step[0] = phi[0, 0] * x_row + phi[0, 1] * kick_row
    step[1] = phi[1, 0] * x_row + phi[1, 1] * kick_row
    step[2] = x_row
    step[3] = np.array([0.0, 0.0, 1.0, 0.0])
    return float(np.max(np.abs(np.linalg.eigvals(step))))


def main() -> int:
    thetas = np.linspace(0.02, 0.6 * math.pi, 600)  # up to 0.3 T, Frost's operating range
    report: dict = {}
    for mu in (0.02, 0.05, 0.1):
        block = {}
        for scheme in ("sync", "lag", "lag_extrap", "lag_adams"):
            radii = np.array([spectral_radius(scheme, t, mu) for t in thetas])
            worst = int(np.argmax(radii))
            block[scheme] = {"max_radius": float(radii[worst]),
                             "at_D_over_T": float(thetas[worst] / (2 * math.pi)),
                             "radius_at_0.22T": float(spectral_radius(scheme, 0.44 * math.pi, mu))}
        # friction (per outer step, gamma D) that makes the lagged scheme stable at 0.22 T
        for scheme in ("lag", "lag_adams"):
            need = None
            for g in np.linspace(0.0, 0.5, 501):
                if spectral_radius(scheme, 0.44 * math.pi, mu, g) <= 1.0 + 1e-12:
                    need = float(g)
                    break
            block[scheme]["gamma_D_needed_at_0.22T"] = need
        report[str(mu)] = block
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    for mu, block in report.items():
        print(f"mu = {mu}")
        for scheme, r in block.items():
            extra = (f"  friction gamma*D needed at 0.22T: {r['gamma_D_needed_at_0.22T']}"
                     if "gamma_D_needed_at_0.22T" in r else "")
            print(f"  {scheme:11s} max radius {r['max_radius']:.5f} at D/T {r['at_D_over_T']:.3f};"
                  f" at 0.22T {r['radius_at_0.22T']:.5f}{extra}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
