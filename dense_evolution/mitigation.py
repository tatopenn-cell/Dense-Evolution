"""
Zero-Noise Extrapolation (ZNE)
-------------------------------
Standard error-mitigation entry points, named the way the field already
names them (Richardson extrapolation, noise factors, zero-noise
extrapolation -- same vocabulary as e.g. Mitiq's `zne` API), so callers
and tooling can find "ZNE" without first learning Dense-Evolution's
internal healing vocabulary.

This module composes `dense_evolution.healing`'s existing primitives
(`calculate_delta_preemp`, ...) -- it does not rename or replace them.
"""
import numpy as np
import jax.numpy as jnp

from .healing import calculate_delta_preemp

__all__ = ["richardson_extrapolate", "zero_noise_extrapolation"]


def richardson_extrapolate(expectation_values, noise_factors) -> jnp.ndarray:
    """Polynomial (Lagrange) Richardson extrapolation to zero noise.

    `expectation_values[i]` is the value measured/simulated at noise scale
    `noise_factors[i]` (e.g. 1x, 2x, 3x folded/scaled noise) -- a scalar,
    or itself an array (e.g. a full probability distribution sampled at
    that noise scale; extrapolated elementwise). Returns the extrapolated
    zero-noise estimate, same shape as one `expectation_values[i]`. Works
    for any number of points and any (not necessarily equally spaced)
    noise factors; for the common 3-point case at noise_factors=(1,2,3)
    this reduces exactly to the textbook coefficients (3, -3, 1).

    `expectation_values` may be complex (e.g. density matrix entries,
    which are complex off-diagonal in general) -- dtype is picked from
    the input itself (complex128 if complex, float64 otherwise, matching
    this function's previous always-float64 behavior for real input
    exactly). Found via a real test case: forcing float64 unconditionally
    silently discarded the imaginary part of complex input with no
    visible error, only a low-signal ComplexWarning easy to miss --
    confirmed directly (`richardson_extrapolate([1+2j, 3+4j], ...)` used
    to return a purely real result, dropping real information).
    """
    lambdas = jnp.asarray(noise_factors, dtype=jnp.float64)
    values_dtype = jnp.complex128 if np.iscomplexobj(np.asarray(expectation_values)) else jnp.float64
    values = jnp.asarray(expectation_values, dtype=values_dtype)
    n = lambdas.shape[0]

    def lagrange_coeff(i):
        others = jnp.concatenate([lambdas[:i], lambdas[i + 1:]])
        return jnp.prod((0.0 - others) / (lambdas[i] - others))

    coeffs = jnp.stack([lagrange_coeff(i) for i in range(n)])
    # Broadcast coeffs against the LEADING axis of values (the "one row per
    # noise scale" axis), not jnp's default trailing-axis alignment --
    # values may itself be array-valued per scale (values.shape = (n,
    # *extra_dims)), e.g. a whole probability distribution rather than a
    # bare scalar. A no-op reshape when values is 1-D (the scalar case),
    # so existing scalar callers are unaffected.
    coeffs = coeffs.reshape((n,) + (1,) * (values.ndim - 1))
    return jnp.sum(coeffs * values, axis=0)


def zero_noise_extrapolation(expectation_values, noise_factors,
                              sigma_at_base_noise=None,
                              target_sigma_ideal: float = 10.0) -> jnp.ndarray:
    """Zero-Noise Extrapolation -- plain, or healing-adapted when a
    coherence signal is available.

    Without `sigma_at_base_noise`: standard Richardson ZNE
    (`richardson_extrapolate`).

    With `sigma_at_base_noise` (the measured/simulated coherence sigma at
    the base, unscaled noise level): the 3 Richardson coefficients are
    perturbed by `dense_evolution.healing.calculate_delta_preemp` -- the
    normalized deviation between the observed sigma and the ideal target
    -- then renormalized to sum to 1. This is Dense-Evolution's
    "predictive healing" ZNE variant: when the observed coherence is off
    the ideal target, the extrapolation is nudged accordingly instead of
    trusting the 3 raw noise-scaled points equally.

    The healing-adapted path currently only supports exactly 3 noise
    factors (the case it has been derived and tested against); passing
    `sigma_at_base_noise` with any other point count raises
    NotImplementedError rather than silently generalizing an unverified
    formula.
    """
    if sigma_at_base_noise is None:
        return richardson_extrapolate(expectation_values, noise_factors)

    lambdas = jnp.asarray(noise_factors, dtype=jnp.float64)
    if lambdas.shape[0] != 3:
        raise NotImplementedError(
            "Healing-adapted ZNE is only defined for exactly 3 noise factors; "
            "call richardson_extrapolate(...) directly for the plain N-point case."
        )
    values_dtype = jnp.complex128 if np.iscomplexobj(np.asarray(expectation_values)) else jnp.float64
    values = jnp.asarray(expectation_values, dtype=values_dtype)
    e_l1, e_l2, e_l3 = values[0], values[1], values[2]

    delta_p = calculate_delta_preemp(jnp.asarray(sigma_at_base_noise, dtype=jnp.float64),
                                      target_sigma_ideal)
    c1 = 3.0 - 0.01 * delta_p
    c2 = -3.0 + 0.02 * delta_p
    c3 = 1.0 - 0.01 * delta_p
    return (c1 * e_l1 + c2 * e_l2 + c3 * e_l3) / (c1 + c2 + c3)
