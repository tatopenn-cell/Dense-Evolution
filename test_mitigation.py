import numpy as np
import jax.numpy as jnp
import pytest

import dense_evolution as de
from dense_evolution.mitigation import (
    richardson_extrapolate, zero_noise_extrapolation, polynomial_extrapolate,
    project_to_physical, uhlmann_fidelity, zne_density_matrix,
)


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


def test_richardson_extrapolate_preserves_complex_input():
    # Regression test: expectation_values used to be forced to jnp.float64
    # unconditionally, silently discarding the imaginary part of complex
    # input (e.g. density-matrix entries) with only a low-signal
    # ComplexWarning. Hand-computed via the textbook (3, -3, 1) Lagrange
    # coefficients on real and imaginary parts separately.
    a, b, c = 1 + 2j, 3 + 4j, 3 + 4j
    d, e, f = 2 + 1j, 6 + 8j, 6 + 8j
    got = np.asarray(richardson_extrapolate([[a, d], [b, e], [c, f]], [1.0, 2.0, 3.0]))
    expected = np.array([
        3.0 * a - 3.0 * b + 1.0 * c,
        3.0 * d - 3.0 * e + 1.0 * f,
    ])
    np.testing.assert_allclose(got, expected, atol=1e-9)
    assert np.iscomplexobj(got)
    assert np.any(np.imag(got) != 0.0) or np.any(np.imag(expected) != 0.0)


def test_richardson_extrapolate_real_input_stays_real():
    got = richardson_extrapolate([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert not np.iscomplexobj(np.asarray(got))


def test_zero_noise_extrapolation_healing_branch_preserves_complex_input():
    e1, e2, e3 = 1 + 2j, 2 + 1j, 3 + 0.5j
    delta_p = abs(9.0 - 10.0) / 10.0
    c1, c2, c3 = 3.0 - 0.01 * delta_p, -3.0 + 0.02 * delta_p, 1.0 - 0.01 * delta_p
    expected = (c1 * e1 + c2 * e2 + c3 * e3) / (c1 + c2 + c3)
    got = complex(zero_noise_extrapolation([e1, e2, e3], [1.0, 2.0, 3.0],
                                            sigma_at_base_noise=9.0,
                                            target_sigma_ideal=10.0))
    assert got == pytest.approx(expected, abs=1e-9)
    assert got.imag != 0.0


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
    assert de.polynomial_extrapolate is polynomial_extrapolate
    assert de.project_to_physical is project_to_physical
    assert de.uhlmann_fidelity is uhlmann_fidelity
    assert de.zne_density_matrix is zne_density_matrix


def test_polynomial_extrapolate_matches_richardson_at_exact_point_count():
    # degree = n_points - 1 means the least-squares fit is exactly
    # determined -- must equal the unique interpolating polynomial, i.e.
    # richardson_extrapolate, to numerical precision.
    rng = np.random.default_rng(10)
    for n in (3, 4, 5):
        values = rng.normal(size=n) + 1j * rng.normal(size=n)
        lambdas = np.arange(1, n + 1, dtype=float)
        expected = np.asarray(richardson_extrapolate(values.tolist(), lambdas.tolist()))
        got = np.asarray(polynomial_extrapolate(values.tolist(), lambdas.tolist(), degree=n - 1))
        np.testing.assert_allclose(got, expected, atol=1e-8)


def test_polynomial_extrapolate_exact_on_matching_degree_polynomial():
    rng = np.random.default_rng(11)
    coeffs = rng.normal(size=3)  # degree-2 polynomial
    lambdas = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # 5 points, overdetermined
    values = sum(c * lambdas ** k for k, c in enumerate(coeffs))
    expected_intercept = coeffs[0]
    got = float(polynomial_extrapolate(values.tolist(), lambdas.tolist(), degree=2))
    assert got == pytest.approx(expected_intercept, abs=1e-6)


def test_polynomial_extrapolate_rejects_underdetermined_fit():
    with pytest.raises(ValueError):
        polynomial_extrapolate([1.0, 2.0], [1.0, 2.0], degree=2)


def test_polynomial_extrapolate_preserves_complex_input():
    values = [1 + 2j, 2 + 1j, 3 + 0.5j, 4 - 1j, 5 - 2j]
    got = polynomial_extrapolate(values, [1.0, 2.0, 3.0, 4.0, 5.0], degree=2)
    assert np.iscomplexobj(np.asarray(got))
    assert complex(got).imag != 0.0


def _random_density_matrix(rng, d):
    a = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    rho = a @ a.conj().T
    return rho / np.trace(rho)


def test_project_to_physical_matches_paper_worked_example():
    # Smolin, Gambetta & Smith (2012), arXiv:1106.5458, Fig. 1: starting
    # eigenvalues 3/5, 1/2, 7/20, 1/10, -11/20 (trace 1, one negative)
    # project to 9/20, 7/20, 1/5, 0, 0 exactly.
    eigvals_in = np.array([3 / 5, 1 / 2, 7 / 20, 1 / 10, -11 / 20])
    assert eigvals_in.sum() == pytest.approx(1.0, abs=1e-12)
    rho_raw = jnp.asarray(np.diag(eigvals_in), dtype=jnp.complex128)

    got = np.asarray(project_to_physical(rho_raw))
    got_eigvals = np.sort(np.linalg.eigvalsh(got))[::-1]
    expected_eigvals = np.array([9 / 20, 7 / 20, 1 / 5, 0.0, 0.0])
    np.testing.assert_allclose(got_eigvals, expected_eigvals, atol=1e-9)


def _random_traceless_hermitian(rng, d):
    a = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    h = (a + a.conj().T) / 2
    return h - (np.trace(h).real / d) * np.eye(d)


def test_project_to_physical_output_is_a_valid_density_matrix():
    rng = np.random.default_rng(2)
    for _ in range(10):
        d = rng.integers(2, 6)
        rho = _random_density_matrix(rng, d)
        # perturb with a traceless Hermitian matrix so trace stays exactly
        # 1 while pushing eigenvalues negative (likely, for this scale).
        rho_raw = jnp.asarray(rho + 0.5 * _random_traceless_hermitian(rng, d), dtype=jnp.complex128)
        got = np.asarray(project_to_physical(rho_raw))
        np.testing.assert_allclose(got, got.conj().T, atol=1e-9)
        assert np.trace(got).real == pytest.approx(1.0, abs=1e-9)
        assert np.trace(got).imag == pytest.approx(0.0, abs=1e-9)
        assert np.linalg.eigvalsh(got).min() >= -1e-9


def test_project_to_physical_is_a_near_no_op_on_already_physical_input():
    rng = np.random.default_rng(3)
    rho = jnp.asarray(_random_density_matrix(rng, 4), dtype=jnp.complex128)
    got = project_to_physical(rho)
    np.testing.assert_allclose(np.asarray(got), np.asarray(rho), atol=1e-9)


def test_uhlmann_fidelity_self_is_one():
    rng = np.random.default_rng(4)
    rho = jnp.asarray(_random_density_matrix(rng, 3), dtype=jnp.complex128)
    assert uhlmann_fidelity(rho, rho) == pytest.approx(1.0, abs=1e-9)


def test_uhlmann_fidelity_matches_pure_state_overlap():
    rng = np.random.default_rng(5)
    for _ in range(5):
        psi_a = rng.normal(size=4) + 1j * rng.normal(size=4)
        psi_a /= np.linalg.norm(psi_a)
        psi_b = rng.normal(size=4) + 1j * rng.normal(size=4)
        psi_b /= np.linalg.norm(psi_b)
        rho_a = jnp.asarray(np.outer(psi_a, psi_a.conj()), dtype=jnp.complex128)
        rho_b = jnp.asarray(np.outer(psi_b, psi_b.conj()), dtype=jnp.complex128)
        expected = abs(np.vdot(psi_a, psi_b)) ** 2
        got = uhlmann_fidelity(rho_a, rho_b)
        assert got == pytest.approx(expected, abs=1e-6)


def test_uhlmann_fidelity_is_symmetric():
    rng = np.random.default_rng(6)
    rho_a = jnp.asarray(_random_density_matrix(rng, 3), dtype=jnp.complex128)
    rho_b = jnp.asarray(_random_density_matrix(rng, 3), dtype=jnp.complex128)
    assert uhlmann_fidelity(rho_a, rho_b) == pytest.approx(uhlmann_fidelity(rho_b, rho_a), abs=1e-9)


def test_zne_density_matrix_output_is_a_valid_density_matrix():
    rng = np.random.default_rng(7)
    d = 4
    rho_at_scales = jnp.stack([
        jnp.asarray(_random_density_matrix(rng, d), dtype=jnp.complex128) for _ in range(3)
    ])
    got = np.asarray(zne_density_matrix(rho_at_scales, [1.0, 2.0, 3.0]))
    np.testing.assert_allclose(got, got.conj().T, atol=1e-9)
    assert np.trace(got).real == pytest.approx(1.0, abs=1e-9)
    assert np.linalg.eigvalsh(got).min() >= -1e-9


def test_zne_density_matrix_is_exactly_richardson_then_projection():
    rng = np.random.default_rng(8)
    d = 3
    rho_at_scales = jnp.stack([
        jnp.asarray(_random_density_matrix(rng, d), dtype=jnp.complex128) for _ in range(3)
    ])
    noise_factors = [1.0, 2.0, 3.0]
    expected = project_to_physical(richardson_extrapolate(rho_at_scales, noise_factors))
    got = zne_density_matrix(rho_at_scales, noise_factors)
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), atol=1e-12)


def test_zne_density_matrix_accepts_more_than_three_scales():
    # Unlike richardson_extrapolate's exact interpolation (which gets
    # measurably worse with more points, see polynomial_extrapolate's
    # docstring), zne_density_matrix's degree=2 least-squares default
    # should stay well-behaved -- still a valid density matrix -- when
    # given more than 3 noise-scale points.
    rng = np.random.default_rng(9)
    d = 3
    rho_at_scales = jnp.stack([
        jnp.asarray(_random_density_matrix(rng, d), dtype=jnp.complex128) for _ in range(6)
    ])
    got = np.asarray(zne_density_matrix(rho_at_scales, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    np.testing.assert_allclose(got, got.conj().T, atol=1e-9)
    assert np.trace(got).real == pytest.approx(1.0, abs=1e-9)
    assert np.linalg.eigvalsh(got).min() >= -1e-9


def test_zne_density_matrix_preserves_complex_off_diagonal_entries():
    # A minimal, hand-built 3-scale complex input where the correct
    # zero-noise extrapolation has a nonzero imaginary part -- guards
    # against the exact bug fixed in richardson_extrapolate resurfacing
    # silently through this composed entry point.
    base = np.array([[0.6, 0.1 + 0.2j], [0.1 - 0.2j, 0.4]])
    rho_at_scales = jnp.stack([
        jnp.asarray(base * (1.0 + 0.1 * s), dtype=jnp.complex128) for s in (0, 1, 2)
    ])
    got = np.asarray(zne_density_matrix(rho_at_scales, [1.0, 2.0, 3.0]))
    assert np.any(np.abs(got.imag) > 1e-9)
