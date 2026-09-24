import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_evolution.noise import pink_noise_p_eff


def test_shape_and_valid_probability_range():
    trace = pink_noise_p_eff(0.05, 256, jax.random.PRNGKey(0))
    assert trace.shape == (256,)
    assert bool((trace >= 0.01).all())
    assert bool((trace <= 0.5).all())


def test_mean_is_close_to_base_p():
    trace = np.array(pink_noise_p_eff(0.05, 4096, jax.random.PRNGKey(1)))
    assert trace.mean() == pytest.approx(0.05, abs=0.01)


def test_different_keys_give_different_traces():
    a = pink_noise_p_eff(0.05, 256, jax.random.PRNGKey(0))
    b = pink_noise_p_eff(0.05, 256, jax.random.PRNGKey(1))
    assert not bool(jnp.allclose(a, b))


def test_power_spectrum_slope_matches_alpha_one_pink_noise():
    # The actual scientific claim: alpha=1.0 must produce a real 1/f
    # spectrum, not just a "noisy-looking" trace. Average the power
    # spectrum over independent realizations (single-realization spectra
    # are individually very noisy -- chi-squared with 2 d.o.f. per bin,
    # by construction of the Timmer & Koenig algorithm) and fit the
    # log-log slope over a mid-frequency band, away from DC and the
    # highest frequencies where finite-sample effects dominate.
    n = 8192
    n_realizations = 30
    psds = []
    for i in range(n_realizations):
        trace = np.array(pink_noise_p_eff(0.05, n, jax.random.PRNGKey(i), amp=1.0))
        psds.append(np.abs(np.fft.rfft(trace - trace.mean())) ** 2)
    psd_mean = np.mean(psds, axis=0)
    freqs = np.fft.rfftfreq(n)

    mask = (freqs > 0.001) & (freqs < 0.3)
    slope, _ = np.polyfit(np.log(freqs[mask]), np.log(psd_mean[mask]), 1)
    assert slope == pytest.approx(-1.0, abs=0.15)


def test_alpha_zero_is_flatter_than_alpha_one():
    # Sanity/contrast check: alpha=0 is white noise (flat spectrum, slope
    # ~0), which must be measurably flatter (less negative slope) than
    # alpha=1's real pink-noise slope -- confirms `alpha` actually controls
    # the spectral shape, not just an arbitrary knob with no effect.
    n = 8192
    n_realizations = 20

    def fitted_slope(alpha):
        psds = []
        for i in range(n_realizations):
            trace = np.array(pink_noise_p_eff(0.05, n, jax.random.PRNGKey(1000 + i), alpha=alpha, amp=1.0))
            psds.append(np.abs(np.fft.rfft(trace - trace.mean())) ** 2)
        psd_mean = np.mean(psds, axis=0)
        freqs = np.fft.rfftfreq(n)
        mask = (freqs > 0.001) & (freqs < 0.3)
        slope, _ = np.polyfit(np.log(freqs[mask]), np.log(psd_mean[mask]), 1)
        return slope

    slope_white = fitted_slope(0.0)
    slope_pink = fitted_slope(1.0)
    assert slope_white > slope_pink + 0.5
