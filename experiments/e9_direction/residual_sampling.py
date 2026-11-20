"""How well can any linear scheme use the teacher's samples? A converse, from measured spectra.

Between teacher calls the multirate loop knows the residual R = F_T - F_S only at
the outer-step samples. Holding it, extrapolating it, or any Kalman or Wiener
predictor built on those samples is a linear estimator of R from its samples.
For a wide-sense stationary R with spectrum S(f), the best such estimator has a
closed form (Matthews, IEEE Trans. Signal Process. 48, 272 (2000)): the aliased
Wiener filter. Its mean squared error at offset j inside an interval of k steps is

    MMSE_j = int over one alias band [ sum_a S_a - |sum_a S_a e^{i w_a j}|^2 / sum_a S_a ] df,

with S_a = S(f + a/k) the aliases that fold onto f. It uses every sample, past and
future, so it bounds every causal scheme from below too.

The same machinery gives a second bound, the one that governs impulse r-RESPA,
whose error is the quadrature error of the momentum impulse int R dt. Among all
linear functionals of the samples, the one closest to the true impulse has
MSE = int [ sum_a |H_a|^2 S_a - |sum_a H_a S_a|^2 / sum_a S_a ] df, where H is the
transfer function of the in-interval integral. The trapezoid rule the impulse uses
is one such functional. Their ratio is its efficiency.

Inputs are E7's measured residual spectra (data/results/e7/residual_profile_*.json,
160 fs periodograms of every atom), stored every 12.5 THz up to 500 THz. The script
first checks the spectral model against E7's own measured staleness (predicted
linear/held ratio against measured), then reports the bounds. No model is run.

Output: data/results/e9/residual_sampling.json

Usage
  LANE=check scripts/safe_run.sh e9_residual_sampling .venvs/frost-mace/bin/python \\
      experiments/e9_direction/residual_sampling.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
E7 = ROOT / "data" / "results" / "e7"
OUT = ROOT / "data" / "results" / "e9" / "residual_sampling.json"
KS = (2, 4, 8, 16)
BINS_PER_BAND = 2000


def spectrum_on_grid(freq_thz, power, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided spectrum on a grid of k * BINS_PER_BAND bins over [-0.5, 0.5) cycles/fs."""
    n = k * BINS_PER_BAND
    f = -0.5 + (np.arange(n) + 0.5) / n
    s = np.interp(np.abs(f) * 1e3, freq_thz, power, right=0.0)
    return f, s


def carry_mse(f, s, k: int) -> dict:
    """Interval-averaged MSE of held, linear-in-time and non-causal LMMSE, as E7 scores them.

    E7 averages over offsets j = 1..k after the sample at t0, and extrapolates
    linearly from the samples at t0 and t0 - k.
    """
    w = 2 * np.pi * f
    held = np.zeros(1)
    lin = np.zeros(1)
    for j in range(1, k + 1):
        held = held + np.sum(s * np.abs(np.exp(1j * w * j) - 1.0) ** 2)
        pred = 1.0 + (j / k) * (1.0 - np.exp(-1j * w * k))
        lin = lin + np.sum(s * np.abs(np.exp(1j * w * j) - pred) ** 2)
    # aliased Wiener filter: fold the grid into k copies of one alias band
    sa = s.reshape(k, BINS_PER_BAND)
    fa = f.reshape(k, BINS_PER_BAND)
    total = sa.sum(0)
    mmse = np.zeros(1)
    for j in range(1, k + 1):
        cross = np.abs((sa * np.exp(1j * 2 * np.pi * fa * j)).sum(0)) ** 2
        mmse = mmse + np.sum(total - cross / np.maximum(total, 1e-300))
    return {"held": float(held[0]) / k, "linear": float(lin[0]) / k, "lmmse": float(mmse[0]) / k}


def impulse_efficiency(f, s, k: int) -> dict:
    """Trapezoid impulse against the best linear functional of the samples, for int_0^k R dt."""
    w = 2 * np.pi * f
    c = np.ones(k + 1)
    c[0] = c[-1] = 0.5  # the true impulse, on the 1 fs frames
    h = sum(c[j] * np.exp(1j * w * j) for j in range(k + 1))
    trap = 0.5 * k * (1.0 + np.exp(1j * w * k))
    mse_trap = float(np.sum(s * np.abs(h - trap) ** 2))
    ha = h.reshape(k, BINS_PER_BAND)
    sa = s.reshape(k, BINS_PER_BAND)
    total = sa.sum(0)
    mmse = float(np.sum((np.abs(ha) ** 2 * sa).sum(0)
                        - np.abs((ha * sa).sum(0)) ** 2 / np.maximum(total, 1e-300)))
    return {"trapezoid": mse_trap, "best_linear": mmse,
            "efficiency": mmse / mse_trap if mse_trap > 0 else float("nan")}


def main() -> int:
    report: dict = {}
    for path in sorted(E7.glob("residual_profile_*.json")):
        data = json.loads(path.read_text())
        run = data["run"]
        report[run] = {}
        for z, entry in data["species"].items():
            if z == "all":
                continue
            freq = np.asarray(entry["spectrum_THz"])
            power = np.asarray(entry["residual_spectrum"])
            rows = {}
            for k in KS:
                f, s = spectrum_on_grid(freq, power, k)
                m = carry_mse(f, s, k)
                imp = impulse_efficiency(f, s, k)
                held_meas = entry["held_force_meV_per_A"][str(k)]
                lin_meas = entry["linear_force_meV_per_A"][str(k)]
                rows[k] = {
                    "linear_over_held_predicted": (m["linear"] / m["held"]) ** 0.5,
                    "linear_over_held_measured": lin_meas / held_meas,
                    "lmmse_over_held": (m["lmmse"] / m["held"]) ** 0.5,
                    "impulse_efficiency": imp["efficiency"],
                }
            report[run][z] = rows
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))

    print("RMS ratios. lin/held: predicted from the spectrum vs measured by E7;")
    print("LMMSE/held: the floor for ANY linear use of the samples (non-causal);")
    print("impulse eff.: best linear impulse MSE / trapezoid MSE (1 = trapezoid is optimal)")
    for run, species in report.items():
        for z, rows in species.items():
            for k, r in rows.items():
                print(f"{run:22s} Z={z:>3} k={k:2d}  lin/held pred {r['linear_over_held_predicted']:.3f}"
                      f" meas {r['linear_over_held_measured']:.3f}   LMMSE/held "
                      f"{r['lmmse_over_held']:.3f}   impulse eff. {r['impulse_efficiency']:.3f}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
