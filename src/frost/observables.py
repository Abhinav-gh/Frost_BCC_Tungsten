"""Physical observables for comparing trajectories and samplers.

Shared by the fidelity harness (E6) and the statistics harness (E8), so both judge
the teacher and Frost by exactly the same estimators.
"""

from __future__ import annotations

import math

import numpy as np
import torch

FS_TO_THZ = 1e3  # 1/fs -> THz


def as_list(values) -> list[float]:
    """JSON-safe list of floats; numpy's float32 is not serialisable."""
    return np.asarray(values, dtype=float).tolist()


def vdos(velocities: torch.Tensor, dt_fs: float,
         max_lag: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Velocity autocorrelation of the whole cell, its spectrum in THz, and the
    correlation itself.

    The correlation is returned because a time step's dispersion error is visible
    there and not in the spectrum. Velocity Verlet shifts a mode of frequency w by
    a fraction (w*dt)^2/24, which is a fixed displacement in frequency -- for
    silica at 2 fs, 0.136 THz against a spectral bin of 0.98 THz, a seventh of one
    bin -- but a phase lag that *accumulates*, reaching eps*t = 8.8 fs by the end
    of a 1024 fs window. The same physics, measured where it grows instead of
    where it is bounded (frost-ec).

    The density of states is the *cosine* transform of the autocorrelation, so the
    one-sided correlation is mirrored into c(-tau) = c(tau) and windowed whole
    before the transform. Taking ``rfft(c * half_window).abs()`` of the one-sided
    correlation instead leaves a flat floor all the way to Nyquist: on a noise-free
    two-mode test signal it puts 26% of the spectrum's power above the highest real
    mode, and on 300 K silica it was 4.7% of the peak, carrying ~37% of the power
    above 40 THz. The mirrored transform puts exactly zero there.
    """
    v = velocities
    n_t = v.shape[0]
    lags = min(max_lag, n_t // 2)
    c = torch.empty(lags, dtype=torch.float64, device=v.device)
    for lag in range(lags):
        c[lag] = (v[: n_t - lag] * v[lag:]).sum(dim=-1).mean()
    c = c / c[0]
    symmetric = torch.cat([c.flip(0)[:-1], c])
    window = torch.hann_window(symmetric.numel(), device=v.device, dtype=torch.float64)
    spectrum = torch.fft.rfft(symmetric * window).real.clamp(min=0.0)  # a density is not negative
    freq = torch.fft.rfftfreq(symmetric.numel(), d=dt_fs) * FS_TO_THZ
    return freq.cpu().numpy(), spectrum.cpu().numpy(), c.cpu().numpy()


def correlation_shift(reference: np.ndarray, other: np.ndarray, interval_fs: float,
                      floor: float = 0.05) -> tuple[float, float]:
    """The fractional frequency shift between two correlations, and its error.

    A dispersion error of fraction eps puts the curves out of step by eps*t, so the
    lag grows linearly with the crossing time and its slope *is* eps -- directly
    comparable with the (w*dt)^2/24 a time step predicts, with no threshold to
    choose.

    Only crossings where the correlation still carries amplitude are used. Silica's
    velocity correlation decays to 0.0014 by the end of a 2048 fs window, so its
    last crossings are noise and a lag read there says nothing; ``floor`` keeps
    crossings whose neighbourhood exceeds that fraction of the curve's peak.

    Returns (eps, standard error), or (nan, nan) when the measurement is
    ambiguous: crossings are paired by time rather than index, because a broad
    band's correlation is a superposition whose oscillations are irregular and a
    crossing present in one run can be absent in the other -- pairing by index then
    compares crossing 8 with crossing 9 and reports tens of femtoseconds where the
    truth is under one. Pairing by time in turn only works while the accumulated
    lag stays below half a period, which is checked.

    Zero crossings need no fitting of the curve itself and are insensitive to
    amplitude, which decays at different rates between runs and would bias a
    correlation-peak fit (frost-ec).
    """
    def crossings(curve: np.ndarray) -> np.ndarray:
        sign = np.sign(curve)
        idx = np.nonzero((sign[:-1] > 0) & (sign[1:] <= 0))[0]
        keep = [i for i in idx if np.abs(curve[max(0, i - 2):i + 3]).max() >= floor * np.abs(curve).max()]
        if not keep:
            return np.empty(0)
        idx = np.asarray(keep)
        frac = curve[idx] / (curve[idx] - curve[idx + 1])
        return (idx + frac) * interval_fs

    a, b = crossings(np.asarray(reference, float)), crossings(np.asarray(other, float))
    if len(a) < 4 or len(b) < 4:
        return float("nan"), float("nan")
    # Pair by time, not by index. A broad band's correlation is a superposition
    # whose oscillations are irregular, so a crossing present in one run can be
    # absent in the other; pairing the i-th of each then compares crossing 8 with
    # crossing 9 and reports a lag of tens of femtoseconds where the truth is
    # under one. Silica did exactly that: +0.65%, +0.08%, then +14.4% and +27.5%.
    # A genuine partner lies within half the local spacing.
    spacing = float(np.median(np.diff(a))) if len(a) > 1 else float("inf")
    pairs = []
    for time in a:
        nearest = b[int(np.argmin(np.abs(b - time)))]
        if abs(nearest - time) <= 0.5 * spacing:
            pairs.append((time, nearest - time))
    if len(pairs) < 4:
        return float("nan"), float("nan")
    # Refuse whenever the two curves hold different numbers of oscillations over
    # their common span. That happens for two reasons and neither is measurable
    # here: the shift has wrapped a whole period (a 3.44% shift over 1024 fs
    # accumulates 35 fs against a 13.9 fs half-period), or the correlation is a
    # broad-band superposition whose oscillations are irregular, so an extra one
    # appears in one run. Silica's teacher shows the second -- 10 crossings against
    # 11 within 584 fs, where its measured 0.558% shift would account for 3.3 fs,
    # an eighth of a period -- and the pairing is contaminated either way. The fit
    # residuals do not see it, so a small standard error there means nothing.
    # The method assumes a quasi-periodic correlation, so check that assumption
    # rather than the result. A clean mode's crossings are evenly spaced; silica's
    # teacher runs 26 to 135 fs about a median of 32, because its correlation is a
    # broad-band superposition whose oscillations are irregular -- an extra one
    # appears in one run and pairing silently compares crossing 8 with crossing 9,
    # reporting tens of femtoseconds where the truth is under one. Fit residuals do
    # not see that, so a small standard error from a contaminated pairing means
    # nothing. Refuse instead.
    for series in (a, b):
        gaps = np.diff(series)
        if len(gaps) < 3:
            return float("nan"), float("nan")
        median = float(np.median(gaps))
        if median <= 0.0 or float(np.abs(gaps - median).max()) > 0.4 * median:
            return float("nan"), float("nan")

    # A lag that grows past half a period pairs each crossing with its neighbour's
    # partner, which shows up as a jump of about one period in the lag sequence where
    # a genuine drift changes by only eps * spacing between pairs. Keep the pairs
    # before the first such jump and fit those: a 3.44% shift over 1024 fs wraps at
    # ~400 fs, and the portion before it still measures the shift. Counting
    # crossings instead is brittle -- over a finite window a shifted curve can gain
    # one crossing right at the boundary for any nonzero shift.
    lags = [pair[1] for pair in pairs]
    for i in range(1, len(lags)):
        if abs(lags[i] - lags[i - 1]) > 0.5 * spacing:
            pairs = pairs[:i]
            break
    if len(pairs) < 4:
        return float("nan"), float("nan")

    t = np.array([p[0] for p in pairs])
    lag = np.array([p[1] for p in pairs])
    n = len(pairs)
    mean_t, mean_lag = t.mean(), lag.mean()
    sxx = float(((t - mean_t) ** 2).sum())
    if sxx <= 0.0:
        return float("nan"), float("nan")
    slope = float(((t - mean_t) * (lag - mean_lag)).sum() / sxx)
    residual = float(((lag - (mean_lag + slope * (t - mean_t))) ** 2).sum())
    error = math.sqrt(residual / ((n - 2) * sxx)) if n > 2 else float("inf")
    return -slope, error  # a positive eps shortens the period, so the lag runs negative


def msd(positions: torch.Tensor, stride_fs: float) -> tuple[np.ndarray, float]:
    """Mean squared displacement over lag time, and the diffusion coefficient in A^2/ps."""
    n_t = positions.shape[0]
    lags = max(n_t // 4, 2)
    out = torch.empty(lags, dtype=torch.float64, device=positions.device)
    for lag in range(lags):
        d = positions[lag:] - positions[: n_t - lag]
        out[lag] = d.pow(2).sum(dim=-1).mean()
    curve = out.cpu().numpy()
    t = np.arange(lags) * stride_fs / 1e3  # ps
    half = slice(lags // 2, lags)
    slope = np.polyfit(t[half], curve[half], 1)[0] if lags > 4 else float("nan")
    return curve, float(slope / 6.0)


def rdf(frames: torch.Tensor, cell: np.ndarray, types: np.ndarray, n_species: int,
        r_max: float = 8.0, bins: int = 160) -> dict:
    """Radial distribution functions, total and per species pair."""
    from frost.neighbours import edges_for

    edges = np.zeros((n_species, n_species, bins))
    total = np.zeros(bins)
    volume = abs(float(np.linalg.det(cell)))
    n_atoms = frames.shape[1]
    for frame in frames.cpu().numpy():
        # the shifts must match the coordinates the distances are taken from: these frames are unwrapped
        # (atoms leave the cell), and shifts built for the wrapped copy gave atoms outside the box wrong
        # distances (8.9 first-shell neighbours instead of fcc's 12 on a 300 K Cu frame; fixed 2026-10-19)
        _, i, j, shifts, _ = edges_for(frame, cell, r_max, dtype=torch.float64, wrap_positions=False)
        r = (torch.as_tensor(frame)[j] - torch.as_tensor(frame)[i] + shifts).norm(dim=1).numpy()
        hist, edges_r = np.histogram(r, bins=bins, range=(0.0, r_max))
        total += hist
        ti, tj = types[i.numpy()], types[j.numpy()]
        for a in range(n_species):
            for b in range(n_species):
                mask = (ti == a) & (tj == b)
                edges[a, b] += np.histogram(r[mask], bins=bins, range=(0.0, r_max))[0]
    centres = 0.5 * (edges_r[1:] + edges_r[:-1])
    shell = 4.0 * np.pi * centres**2 * (edges_r[1] - edges_r[0])
    norm = frames.shape[0] * n_atoms * shell * (n_atoms / volume)
    counts = np.bincount(types, minlength=n_species)
    pairs = {}
    for a in range(n_species):
        for b in range(n_species):
            if counts[a] and counts[b]:
                pairs[f"{a}{b}"] = as_list(edges[a, b] / (frames.shape[0] * counts[a] * shell
                                                          * (counts[b] / volume)))
    return {"r": as_list(centres), "g_total": as_list(total / norm), "g_pairs": pairs}


def tau_int(series, c: float = 5.0) -> tuple[float, float]:
    """Integrated autocorrelation time in samples, and its standard error.

    Sokal's self-consistent window: the smallest M with M >= c * tau(M). The error
    is Madras and Sokal's (1988) large-N variance, var(tau) ~ (4M + 2) / N * tau^2.
    It is large: a chain only a few hundred tau long knows its own tau to tens of
    percent, and every ESS ratio inherits that.
    """
    x = np.asarray(series, dtype=float) - np.mean(series)
    n = len(x)
    if n < 8 or not np.any(x):
        return float("nan"), float("nan")
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n].real
    rho = acf / acf[0]
    tau, window = 1.0, n - 1
    for m in range(1, n):
        tau = 1.0 + 2.0 * rho[1:m + 1].sum()
        if m >= c * tau:
            window = m
            break
    tau = max(tau, 1.0)
    return float(tau), float(tau * np.sqrt((4 * window + 2) / n))


def ratio_se(a: float, a_se: float, b: float, b_se: float) -> float:
    """Standard error of a / b for independent a and b, to first order."""
    return float(abs(a / b) * np.hypot(a_se / a, b_se / b))


def fastest_period_fs(freq_THz, spectrum, quantile: float = 0.99) -> float:
    """The period of the fastest mode carrying real weight, in fs.

    The certificate bounds the outer step at a fraction of this, so the convention
    that sets it decides the interval. An eyeballed high-frequency edge is not
    reproducible across systems, and comparing systems is the whole point of the
    bound, so the edge is the frequency below which ``quantile`` of the vibrational
    density of states lies. Report the quantile with the period: at 0.98 or 0.995
    the answer moves, and a table mixing conventions is not a table.

    Note that this is a spectral convention, not a claim about a particular bond:
    silica's ~30 fs and Li3PO4's ~33 fs were read off Si-O and P-O stretch
    frequencies instead, which is why every period should be re-derived this way
    before the fractions in plan section 1.14 are compared across systems.
    """
    f = np.asarray(freq_THz, dtype=float)
    p = np.clip(np.asarray(spectrum, dtype=float), 0.0, None)
    if f.size < 2 or not p.any():
        return float("nan")
    cumulative = np.cumsum(p) / p.sum()
    edge = float(f[int(np.searchsorted(cumulative, quantile))]) if cumulative[-1] >= quantile else float(f[-1])
    return float("inf") if edge <= 0.0 else 1e3 / edge  # THz -> fs
