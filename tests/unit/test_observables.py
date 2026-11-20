

def test_fastest_period_tracks_the_high_frequency_edge_not_the_peak():
    """The cap is a fraction of this, so it must follow the fastest real mode."""
    import numpy as np

    from frost.observables import fastest_period_fs

    freq = np.linspace(0.0, 150.0, 600)
    slow_only = np.exp(-(((freq - 10.0) / 4.0) ** 2))
    with_stretch = slow_only + 0.7 * np.exp(-(((freq - 33.0) / 3.0) ** 2))

    # a band at 33 THz shortens the period even though the peak stays at 10 THz
    assert fastest_period_fs(freq, with_stretch) < fastest_period_fs(freq, slow_only)
    # and the convention is monotone in the quantile, so the quantile must be quoted
    periods = [fastest_period_fs(freq, with_stretch, q) for q in (0.98, 0.99, 0.995)]
    assert periods[0] > periods[1] > periods[2]


def test_fastest_period_is_defined_on_degenerate_input():
    import numpy as np

    from frost.observables import fastest_period_fs

    freq = np.linspace(0.0, 150.0, 10)
    assert np.isnan(fastest_period_fs(freq, np.zeros_like(freq)))
    assert np.isnan(fastest_period_fs(np.array([1.0]), np.array([1.0])))


def test_vdos_puts_no_power_above_the_highest_real_mode():
    """A one-sided ACF transformed with a half window leaves a floor to Nyquist."""
    import numpy as np
    import torch

    from frost.observables import vdos

    dt, n_t, n_atoms = 2.0, 8192, 120
    t = np.arange(n_t) * dt
    rng = np.random.default_rng(0)
    phase = rng.uniform(0, 2 * np.pi, (n_atoms, 2, 3))
    signal = (np.cos(2 * np.pi * 10e-3 * t)[None, :, None] * np.cos(phase[:, 0:1, :])
              + 0.6 * np.cos(2 * np.pi * 33e-3 * t)[None, :, None] * np.cos(phase[:, 1:2, :]))
    v = torch.as_tensor(signal.transpose(1, 0, 2))  # noise-free: any floor is an artefact

    freq, spectrum, _ = vdos(v, dt, 1024)
    spectrum = spectrum / spectrum.max()
    above = freq > 45.0  # both modes are below this
    assert spectrum[above].sum() / spectrum.sum() < 1e-3
    assert spectrum[above].max() < 1e-3
    # and the two real modes are still there
    for mode in (10.0, 33.0):
        near = np.abs(freq - mode) < 2.0
        assert spectrum[near].max() > 0.1


def test_vdos_frequencies_do_not_depend_on_the_integration_step():
    """A sampling interval mislabelled as --sample-every scales every frequency.

    validate.py sampled every `max(round(sample_every/dt), 1)` steps and then told
    vdos() the interval was `sample_every`. Once dt exceeds sample_every the stride
    floors at one step and the two disagree: Cu's first moment read 9.498 THz at
    dt = 4 against 4.750 at dt = 1, a factor of 1.9996, which is a time axis and
    not physics.
    """
    import numpy as np
    import pytest
    import torch

    from frost.observables import vdos

    n_t, n_atoms = 4096, 80
    rng = np.random.default_rng(0)
    phase = rng.uniform(0, 2 * np.pi, (n_atoms, 3))

    def spectrum_at(interval_fs: float):
        t = np.arange(n_t) * interval_fs
        signal = np.cos(2 * np.pi * 12e-3 * t)[None, :, None] * np.cos(phase[:, None, :])
        v = torch.as_tensor(signal.transpose(1, 0, 2))
        freq, power, _ = vdos(v, interval_fs, 512)
        return float(freq[int(np.argmax(power))])

    # the same 12 THz motion, sampled at two intervals, must report 12 THz twice
    assert spectrum_at(2.0) == pytest.approx(12.0, abs=0.5)
    assert spectrum_at(4.0) == pytest.approx(12.0, abs=0.5)


def test_correlation_shift_recovers_a_known_frequency_shift():
    """The lag grows as eps*t, so its slope across crossings is eps itself."""
    import numpy as np
    import pytest

    from frost.observables import correlation_shift

    interval, n = 2.0, 512
    t = np.arange(n) * interval
    period = 27.7  # silica's fastest mode
    reference = np.cos(2 * np.pi * t / period)
    for eps in (0.0086, 0.0344):  # silica at dt = 2 and dt = 4
        # the method is unambiguous only while eps * t stays under half a period,
        # so the larger shift is measured over a correspondingly shorter window
        span = int(min(n, 0.45 * period / (eps * interval)))
        shifted = np.cos(2 * np.pi * (1 + eps) * t / period)
        measured, error = correlation_shift(reference[:span], shifted[:span], interval)
        assert measured == pytest.approx(eps, rel=0.15), eps
        assert error < 0.2 * eps  # and the fit knows it


def test_correlation_shift_uses_only_the_unambiguous_portion_of_a_wrapping_lag():
    """Past half a period the lag wraps; the portion before the wrap still measures it."""
    import numpy as np
    import pytest

    from frost.observables import correlation_shift

    interval, n, period = 2.0, 512, 27.7
    t = np.arange(n) * interval
    reference = np.cos(2 * np.pi * t / period)
    far = np.cos(2 * np.pi * 1.0344 * t / period)  # 35 fs of lag by the end, 13.9 half-period
    measured, _ = correlation_shift(reference, far, interval)
    assert measured == pytest.approx(0.0344, rel=0.15)


def test_correlation_shift_refuses_a_broad_band_correlation():
    """An irregular superposition gives unpaired crossings; refuse rather than guess."""
    import numpy as np

    from frost.observables import correlation_shift

    interval = 2.0
    t = np.arange(512) * interval
    rng = np.random.default_rng(0)
    band = sum(np.cos(2 * np.pi * t / p + rng.uniform(0, 6.3)) for p in (22.0, 27.7, 35.0, 48.0))
    measured, error = correlation_shift(band, band * 1.0 + 0.0, interval)
    assert np.isnan(measured) and np.isnan(error)
