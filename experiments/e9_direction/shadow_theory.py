"""A shadow-Hamiltonian drift monitor for impulse r-RESPA, from data the loop already has.

One outer step of Frost is the Strang splitting  e^{D/2 B} e^{D A} e^{D/2 B}:
B = U_R = U_T - U_S is the residual potential, kicked at both ends, and
A = K + U_S is the student Hamiltonian, whose flow the inner steps follow. A
symplectic splitting conserves a shadow Hamiltonian exactly (to all orders in
backward error analysis), and its leading terms are Poisson brackets:

    H~ = H + D^2 ( a {B,{B,A}} + b {A,{A,B}} ) + O(D^4),        H = A + B = H_T.

For MD both brackets are available without an extra model call:
    {B,{B,A}} = R . M^-1 R                  (the residual, in acceleration units)
    {A,{A,B}} = d^2 U_R / dt^2 along the student flow
              = v^T H_R v - R . M^-1 F_S,
and the second derivative is a cubic-Hermite difference of the teacher and
student energies and forces at the two ends of an outer step, which the loop
computes anyway.

Lattice QCD tunes nested (multi-timescale) HMC integrators with exactly these
brackets (Clark, Joo, Kennedy, Silva 2011; Kennedy, Silva, Clark 2013); Engle,
Skeel and Drees (2005) monitor MD energy drift with shadows. Here the question is
whether H~ is quiet enough to make Frost's drift clause decide sooner.

Checks, on a Morse teacher with a harmonic student (mu = 0 and mu = -0.2):
1. the coefficients (a, b), fitted by least squares, against theory;
2. the fluctuation of H and of H~ (exact brackets, and brackets from Hermite
   differences of outer-step data only) at stable outer steps;
3. the D^2 scaling of the fluctuation that H~ removes.

Output: data/results/e9/shadow_theory.json

Usage
  LANE=check scripts/safe_run.sh e9_shadow_theory .venvs/frost-mace/bin/python \\
      experiments/e9_direction/shadow_theory.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "results" / "e9" / "shadow_theory.json"


def make(mu: float):
    """Teacher V_T = (1 - e^-x)^2; student V_S = (1 + mu) x^2 (curvature 2 (1 + mu))."""
    ks = 2.0 * (1.0 + mu)

    def u_t(x):
        return (1.0 - np.exp(-x)) ** 2

    def f_t(x):
        ex = np.exp(-x)
        return -2.0 * (1.0 - ex) * ex

    def u_s(x):
        return 0.5 * ks * x * x

    def f_s(x):
        return -ks * x

    def ur2(x):  # U_R'' exactly
        ex = np.exp(-x)
        return 2.0 * ex * (2.0 * ex - 1.0) - ks

    return ks, u_t, f_t, u_s, f_s, ur2


def run(mu: float, big: float, energy: float, periods: int = 300):
    ks, u_t, f_t, u_s, f_s, ur2 = make(mu)
    w = math.sqrt(ks)
    c, s = math.cos(w * big), math.sin(w * big)
    x = -math.log(1.0 + math.sqrt(energy))
    p = 0.0
    steps = int(periods * 2 * math.pi / w / big)
    rows = []
    for _ in range(steps):
        r0 = f_t(x) - f_s(x)
        p_in = p + 0.5 * big * r0  # after the first half kick: the student flow starts here
        x1, p1 = c * x + s / w * p_in, -w * s * x + c * p_in  # exact harmonic student flow
        r1 = f_t(x1) - f_s(x1)
        # outer-step data only: energies and forces at both ends, velocities inside
        ur0, ur1 = u_t(x) - u_s(x), u_t(x1) - u_s(x1)
        d0, d1 = -r0 * p_in, -r1 * p1  # dU_R/dt along the student flow at both ends
        hermite_end = (-6.0 * (ur1 - ur0) + big * (2.0 * d0 + 4.0 * d1)) / big**2
        p = p1 + 0.5 * big * r1
        x = x1
        h = 0.5 * p * p + u_t(x)
        pbb = r1 * r1  # {B,{B,A}} = |R|^2
        paa = p * p * ur2(x) - (-f_s(x)) * (-r1)  # {A,{A,B}} = p^2 U_R'' - U_S' U_R'
        rows.append((h, pbb, paa, hermite_end + 0.0 * p))
        if not math.isfinite(h) or abs(h - energy) > 1.0:
            return None
    return np.asarray(rows)


def main() -> int:
    report: dict = {}
    for mu in (0.0, -0.2):
        block: dict = {"rows": []}
        # 1. fit (a, b) at a small stable step, where the O(D^4) terms are negligible
        rows = run(mu, 0.05, 0.05)
        h, pbb, paa = rows[:, 0], rows[:, 1], rows[:, 2]
        design = np.column_stack([pbb - pbb.mean(), paa - paa.mean()]) * 0.05**2
        coef, *_ = np.linalg.lstsq(design, -(h - h.mean()), rcond=None)
        block["fitted_a_b"] = [float(coef[0]), float(coef[1])]
        a, b = coef
        # 2-3. fluctuation of H, of H~ with exact brackets, and of H~ from outer-step data
        for frac in (0.05, 0.10, 0.15, 0.20):
            big = frac * 2 * math.pi / math.sqrt(2.0 * (1.0 + mu))
            rows = run(mu, big, 0.05)
            if rows is None:
                block["rows"].append({"D_over_T": frac, "stable": False})
                continue
            h, pbb, paa, herm = rows.T
            exact = h + big**2 * (a * pbb + b * paa)
            # in the data-only version the bracket is d^2 U_R/dt^2 at the step's end,
            # evaluated with the pre-kick momentum; the O(D) mismatch is O(D^3) in H~
            data_only = h + big**2 * (a * pbb + b * herm)
            sd = float(np.std(h))
            block["rows"].append({
                "D_over_T": frac, "stable": True,
                "sd_H": sd,
                "sd_shadow_exact": float(np.std(exact)),
                "sd_shadow_from_outer_data": float(np.std(data_only)),
                "reduction_exact": sd / float(np.std(exact)),
                "reduction_from_outer_data": sd / float(np.std(data_only)),
            })
        report[str(mu)] = block
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    for mu, block in report.items():
        print(f"mu = {mu}: fitted (a, b) = ({block['fitted_a_b'][0]:.5f}, {block['fitted_a_b'][1]:.5f})"
              f"   1/24 = {1 / 24:.5f}, 1/12 = {1 / 12:.5f}")
        for r in block["rows"]:
            if not r["stable"]:
                print(f"  D/T {r['D_over_T']:.2f}: unstable")
                continue
            print(f"  D/T {r['D_over_T']:.2f}: sd(H) {r['sd_H']:.3e}  sd(H~ exact) "
                  f"{r['sd_shadow_exact']:.3e} (x{r['reduction_exact']:.0f})  sd(H~ from outer data) "
                  f"{r['sd_shadow_from_outer_data']:.3e} (x{r['reduction_from_outer_data']:.1f})")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
