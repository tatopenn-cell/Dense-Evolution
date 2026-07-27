import numpy as np
import pytest

import dense_evolution as de
from dense_evolution.mitigation import richardson_extrapolate, zero_noise_extrapolation


def test_richardson_extrapolate_matches_known_3point_coefficients():
    rng = np.random.default_rng(0)
    for _ in range(20):
        e1, e2, e3 = rng.normal(size=3)
        expected = 3.0 * e1 - 3.0 * e2 + 1.0 * e3
        got = float(richardson_extrapolate([e1, e2, e3], [1.0, 2.0, 3.0]))
        assert got == pytest.approx(expected, abs=1e-9)


def test_richardson_extrapolate_exact_on_linear_data():
    # a Richardson/Lagrange extrapolation is exact for any polynomial of
    # degree < n_points; for 3 points a linear signal must extrapolate
    # to exactly the intercept.
    a, b = 5.3, -2.1
    lambdas = [1.0, 2.0, 3.0]
    values = [a + b * l for l in lambdas]
    got = float(richardson_extrapolate(values, lambdas))
    assert got == pytest.approx(a, abs=1e-9)


def test_richardson_extrapolate_supports_vector_valued_expectation_values():
    # expectation_values[i] doesn't have to be a scalar -- e.g. a full
    # probability distribution sampled at noise scale i. Found via a real
    # broadcasting bug: coeffs (shape (n,)) times a stacked (n, d) array
    # relies on jnp's default trailing-axis alignment, which pairs (n,)
    # against d, not n -- fails outright unless d happens to equal n.
    rng = np.random.default_rng(1)
    v1, v2, v3 = rng.normal(size=4), rng.normal(size=4), rng.normal(size=4)
    got = np.asarray(richardson_extrapolate([v1, v2, v3], [1.0, 2.0, 3.0]))
    expected = 3.0 * v1 - 3.0 * v2 + 1.0 * v3
    np.testing.assert_allclose(got, expected, atol=1e-9)
    assert got.shape == (4,)


def test_zero_noise_extrapolation_without_sigma_matches_richardson_extrapolate():
    values, lambdas = [1.234, 0.876, 0.611], [1.0, 2.0, 3.0]
    plain = float(richardson_extrapolate(values, lambdas))
    orchestrated = float(zero_noise_extrapolation(values, lambdas))
    assert orchestrated == pytest.approx(plain, abs=1e-12)


def test_zero_noise_extrapolation_with_sigma_matches_reference_healing_formula():
    # reference formula, promoted verbatim from
    # Dense-Evolution-Ising-Tests/tests/test_zne_predictive_healing.py
    # (_adaptive_healing_richardson), independently re-derived here.
    def reference(e_l1, e_l2, e_l3, delta_p):
        c1, c2, c3 = 3.0 - 0.01 * delta_p, -3.0 + 0.02 * delta_p, 1.0 - 0.01 * delta_p
        return (c1 * e_l1 + c2 * e_l2 + c3 * e_l3) / (c1 + c2 + c3)

    target_sigma = 10.0
    for sigma, (e1, e2, e3) in [
        (9.5, (1.0, 0.8, 0.6)),
        (7.0, (2.3, 1.9, 1.5)),
        (10.0, (-0.4, -0.5, -0.6)),
    ]:
        delta_p = abs(sigma - target_sigma) / target_sigma
        expected = reference(e1, e2, e3, delta_p)
        got = float(zero_noise_extrapolation([e1, e2, e3], [1.0, 2.0, 3.0],
                                              sigma_at_base_noise=sigma,
                                              target_sigma_ideal=target_sigma))
        assert got == pytest.approx(expected, abs=1e-9)


def test_zero_noise_extrapolation_rejects_non_3point_healing_request():
    with pytest.raises(NotImplementedError):
        zero_noise_extrapolation([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0],
                                  sigma_at_base_noise=9.0)


def test_exported_from_package_root():
    assert de.richardson_extrapolate is richardson_extrapolate
    assert de.zero_noise_extrapolation is zero_noise_extrapolation
