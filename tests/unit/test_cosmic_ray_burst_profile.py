import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from dense_evolution import cosmic_ray_burst_profile


def test_at_impact_returns_baseline():
    out = cosmic_ray_burst_profile(jnp.array([0.0]), baseline_gamma=0.05)
    assert float(out[0]) == 0.05


def test_recovers_to_baseline_after_many_decay_constants():
    # 6 decay constants (25ms default) -> essentially fully recovered.
    out = cosmic_ray_burst_profile(jnp.array([150000.0]), baseline_gamma=0.05)
    assert abs(float(out[0]) - 0.05) / 0.05 < 0.01


def test_reaches_near_peak_ratio_before_decaying_away():
    # At 1ms both default rise stages (tau1=3us, tau2=300us) are long
    # saturated, while the 25ms decay has barely acted.
    out = cosmic_ray_burst_profile(jnp.array([1000.0]), baseline_gamma=0.05)
    assert abs(float(out[0]) - 0.05 * 3.75) < 0.05 * 0.2


def test_scales_linearly_with_baseline_gamma():
    t = jnp.array([0.0, 10.0, 1000.0, 50000.0])
    out1 = cosmic_ray_burst_profile(t, baseline_gamma=0.05)
    out2 = cosmic_ray_burst_profile(t, baseline_gamma=0.10)
    assert np.allclose(np.array(out2), np.array(out1) * 2.0)


def test_custom_ratios_and_timescales_are_respected():
    out_default = cosmic_ray_burst_profile(jnp.array([1000.0]), baseline_gamma=0.05)
    out_custom = cosmic_ray_burst_profile(jnp.array([1000.0]), baseline_gamma=0.05,
                                           ratio_peak=10.0, ratio_intermediate=5.0)
    assert float(out_custom[0]) > float(out_default[0])


def test_output_shape_matches_input():
    t = jnp.linspace(0.0, 1000.0, 37)
    out = cosmic_ray_burst_profile(t, baseline_gamma=0.05)
    assert out.shape == t.shape
