"""A noise strength that oscillates instead of scaling smoothly/linearly
-- for stress-testing mitigation techniques whose extrapolation assumes a
smooth noise-vs-scale relationship."""
import jax.numpy as jnp

__all__ = ["oscillating_p_eff"]


def oscillating_p_eff(base_p: float, factor: float, freq: float, amp: float) -> jnp.ndarray:
    """Effective noise probability that oscillates around `base_p` as a
    function of `factor` (e.g. a ZNE noise-scale factor), instead of
    scaling smoothly with it: `base_p * (1 + amp * sin(factor * pi /
    freq))`, clipped to `[0.01, 0.5]` so it always stays a valid
    probability.

    Promoted from Dense-Evolution-Discovery's jsd_zne_oscillating_noise.py,
    where it was used to build a noise-vs-scale relationship deliberately
    NOT smooth/monotonic, to stress-test
    `dense_evolution.mitigation.jsd_predictive_zne_density_matrix` against
    noise models where plain Richardson extrapolation's smoothness
    assumption breaks down.

    Examples
    --------
    >>> from dense_evolution.noise import oscillating_p_eff
    >>> round(float(oscillating_p_eff(base_p=0.1, factor=0.0, freq=2.0, amp=0.5)), 4)
    0.1
    >>> round(float(oscillating_p_eff(base_p=0.1, factor=1.0, freq=2.0, amp=0.5)), 4)
    0.15
    """
    p = base_p * (1.0 + amp * jnp.sin(factor * jnp.pi / freq))
    return jnp.clip(p, 0.01, 0.5)
