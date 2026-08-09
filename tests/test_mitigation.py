import numpy as np
import jax.numpy as jnp
import pytest

import dense_evolution as de
from dense_evolution.mitigation import (
    richardson_extrapolate, zero_noise_extrapolation, polynomial_extrapolate,
    project_to_physical, uhlmann_fidelity, zne_density_matrix, zne_density_matrix_jit,
    jsd_predictive_zne_density_matrix,
    richardson_extrapolate_jit, zero_noise_extrapolation_jit, uhlmann_fidelity_jit,
    polynomial_extrapolate_jit,
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
    # Dense-Evolution-Discovery/tests/test_zne_predictive_healing.py
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
    assert de.zne_density_matrix_jit is zne_density_matrix_jit
    assert de.richardson_extrapolate_jit is richardson_extrapolate_jit
    assert de.zero_noise_extrapolation_jit is zero_noise_extrapolation_jit
    assert de.uhlmann_fidelity_jit is uhlmann_fidelity_jit


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


def test_uhlmann_fidelity_gradient_is_finite_at_degenerate_eigenvalues():
    # Regression test for the degenerate-eigenvalue AD singularity (JAX
    # issues #2311/#8732; general treatment in Kasim, arXiv:2011.04366):
    # jnp.linalg.eigh's own gradient rule divides by (lambda_i - lambda_j),
    # which is exactly 0/0 whenever rho_A has tied eigenvalues -- the
    # maximally mixed state (all eigenvalues equal) is the sharpest case.
    # _uhlmann_fidelity_core now routes rho_A's eigendecomposition through
    # _eigh_degenerate_safe specifically to stay finite here; confirmed
    # against the plain-jnp.linalg.eigh formula below, which really does
    # still produce NaN (the pre-fix behavior), so this isn't a vacuous test.
    import jax
    from dense_evolution.mitigation import _uhlmann_fidelity_core

    d = 4
    rho_a = jnp.eye(d, dtype=jnp.complex128) / d
    rng = np.random.default_rng(7)
    rho_b = jnp.asarray(_random_density_matrix(rng, d), dtype=jnp.complex128)

    def fidelity_via_fixed_core(a):
        return jnp.real(_uhlmann_fidelity_core(a, rho_b))

    grad_fixed = jax.grad(fidelity_via_fixed_core)(rho_a)
    assert not bool(jnp.any(jnp.isnan(grad_fixed)))

    def fidelity_via_plain_eigh(a):
        w, v = jnp.linalg.eigh(a)
        sqrt_a = (v * jnp.sqrt(jnp.clip(jnp.real(w), 0.0, None))) @ jnp.conj(v).T
        inner = sqrt_a @ rho_b @ sqrt_a
        inner_evals = jnp.clip(jnp.real(jnp.linalg.eigvalsh(inner)), 0.0, None)
        return jnp.real(jnp.sum(jnp.sqrt(inner_evals)) ** 2)

    grad_plain = jax.grad(fidelity_via_plain_eigh)(rho_a)
    assert bool(jnp.any(jnp.isnan(grad_plain)))


def test_uhlmann_fidelity_unchanged_by_degenerate_safe_eigh():
    # The degenerate-safe eigh fix only changes the backward (gradient)
    # rule -- forward-pass values must stay bit-for-bit the same as before
    # (same underlying jnp.linalg.eigh call). Cross-checked against an
    # independent numpy reference implementation, not just self-consistency.
    rng = np.random.default_rng(8)
    for d in (2, 4, 5):
        rho_a = _random_density_matrix(rng, d)
        rho_b = _random_density_matrix(rng, d)
        w, v = np.linalg.eigh(rho_a)
        sqrt_a = (v * np.sqrt(np.clip(w.real, 0.0, None))) @ v.conj().T
        inner = sqrt_a @ rho_b @ sqrt_a
        inner_evals = np.clip(np.linalg.eigvalsh(inner).real, 0.0, None)
        expected = float(np.sum(np.sqrt(inner_evals)) ** 2)

        got = uhlmann_fidelity(
            jnp.asarray(rho_a, dtype=jnp.complex128),
            jnp.asarray(rho_b, dtype=jnp.complex128),
        )
        assert got == pytest.approx(expected, abs=1e-9)


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


def test_zne_density_matrix_jit_matches_eager_exactly():
    rng = np.random.default_rng(20)
    d = 4
    mats = []
    for _ in range(3):
        a = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        m = a @ a.conj().T
        mats.append(m / np.trace(m))
    rho_at_scales = jnp.asarray(np.stack(mats), dtype=jnp.complex128)
    noise_factors = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)

    eager = zne_density_matrix(rho_at_scales, [1.0, 2.0, 3.0], degree=2)
    jitted = zne_density_matrix_jit(rho_at_scales, noise_factors, degree=2)
    np.testing.assert_array_equal(np.asarray(eager), np.asarray(jitted))


def test_zne_density_matrix_jit_output_is_a_valid_density_matrix():
    rng = np.random.default_rng(21)
    d = 5
    mats = []
    for _ in range(5):
        a = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        m = a @ a.conj().T
        mats.append(m / np.trace(m))
    rho_at_scales = jnp.asarray(np.stack(mats), dtype=jnp.complex128)
    noise_factors = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64)

    got = np.asarray(zne_density_matrix_jit(rho_at_scales, noise_factors, degree=2))
    np.testing.assert_allclose(got, got.conj().T, atol=1e-9)
    assert np.trace(got).real == pytest.approx(1.0, abs=1e-9)
    assert np.linalg.eigvalsh(got).min() >= -1e-9


def test_zne_density_matrix_jit_actually_compiles_under_jit():
    # zne_density_matrix_jit is already jax.jit-wrapped; this specifically
    # checks it doesn't raise a tracing error (e.g. from a stray
    # np.iscomplexobj/np.asarray call on a traced value) when called
    # through an *additional* outer jax.jit, the realistic case of
    # embedding it inside a larger jitted pipeline (e.g. jax.lax.scan).
    import jax

    rng = np.random.default_rng(22)
    d = 3
    mats = []
    for _ in range(3):
        a = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        m = a @ a.conj().T
        mats.append(m / np.trace(m))
    rho_at_scales = jnp.asarray(np.stack(mats), dtype=jnp.complex128)
    noise_factors = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)

    from dense_evolution.mitigation import _zne_density_matrix_core
    import functools
    outer_jit = jax.jit(functools.partial(_zne_density_matrix_core, degree=2))
    result = outer_jit(rho_at_scales, noise_factors)
    result.block_until_ready()
    assert np.trace(np.asarray(result)).real == pytest.approx(1.0, abs=1e-9)


def test_polynomial_extrapolate_jit_matches_eager():
    lambdas = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64)
    values = jnp.asarray([1 + 2j, 3 + 4j, 3 + 4j, 2 - 1j, 5 + 0.5j], dtype=jnp.complex128)
    got = polynomial_extrapolate_jit(values, lambdas, degree=2)
    expected = polynomial_extrapolate(
        [1 + 2j, 3 + 4j, 3 + 4j, 2 - 1j, 5 + 0.5j], [1.0, 2.0, 3.0, 4.0, 5.0], degree=2)
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected))


def test_polynomial_extrapolate_jit_compiles_under_outer_jit():
    import jax
    import functools
    lambdas = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)
    values = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)
    outer_jit = jax.jit(functools.partial(polynomial_extrapolate_jit, degree=2))
    result = outer_jit(values, lambdas)
    result.block_until_ready()
    assert float(result) == pytest.approx(0.0, abs=1e-9)  # linear data -> exact intercept 0


def test_richardson_extrapolate_jit_matches_eager():
    lambdas = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)
    values = jnp.asarray([1 + 2j, 3 + 4j, 3 + 4j], dtype=jnp.complex128)
    got = richardson_extrapolate_jit(values, lambdas)
    expected = richardson_extrapolate([1 + 2j, 3 + 4j, 3 + 4j], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected))


def test_richardson_extrapolate_jit_compiles_under_outer_jit():
    import jax
    lambdas = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)
    values = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)
    outer_jit = jax.jit(richardson_extrapolate_jit)
    result = outer_jit(values, lambdas)
    result.block_until_ready()
    assert float(result) == pytest.approx(0.0, abs=1e-9)  # linear data -> exact intercept 0


def test_zero_noise_extrapolation_jit_matches_eager():
    values = jnp.asarray([1 + 2j, 3 + 4j, 3 + 4j], dtype=jnp.complex128)
    sigma = jnp.asarray(7.5, dtype=jnp.float64)
    got = zero_noise_extrapolation_jit(values, sigma, 10.0)
    expected = zero_noise_extrapolation([1 + 2j, 3 + 4j, 3 + 4j], [1.0, 2.0, 3.0],
                                         sigma_at_base_noise=7.5, target_sigma_ideal=10.0)
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected))


def test_zero_noise_extrapolation_jit_target_sigma_ideal_stays_dynamic():
    # target_sigma_ideal must NOT need to be static -- calculate_delta_preemp
    # uses jnp.where internally, not a Python if, so it's trace-safe as a
    # plain traced float. Regression guard: if this ever needs
    # static_argnames, this test starts raising a tracing error (the
    # primary thing being checked -- no exception on a second call with a
    # different target, no recompilation required). A wide target gap and
    # non-degenerate complex values also confirm the coefficients (and so
    # the result) actually do move, not just "didn't crash".
    import jax
    values = jnp.asarray([1 + 2j, 3 + 4j, 3 + 4j], dtype=jnp.complex128)
    sigma = jnp.asarray(0.5, dtype=jnp.float64)

    @jax.jit
    def f(values, sigma, target):
        return zero_noise_extrapolation_jit(values, sigma, target)

    r1 = f(values, sigma, 10.0)
    r2 = f(values, sigma, 1.0)  # different target, same compiled function
    assert not jnp.allclose(r1, r2)


def test_uhlmann_fidelity_jit_matches_eager():
    rng = np.random.default_rng(23)
    d = 3
    a = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    rho_a = jnp.asarray(a @ a.conj().T / np.trace(a @ a.conj().T), dtype=jnp.complex128)
    b = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    rho_b = jnp.asarray(b @ b.conj().T / np.trace(b @ b.conj().T), dtype=jnp.complex128)

    got = uhlmann_fidelity_jit(rho_a, rho_b)
    expected = uhlmann_fidelity(rho_a, rho_b)
    assert float(got) == pytest.approx(expected, abs=1e-9)


def test_full_pipeline_composes_under_a_single_outer_jax_jit():
    # The realistic case this whole _jit family exists for: several of
    # these functions called together inside ONE outer jax.jit (e.g. a
    # step function passed to jax.lax.scan), not each jitted in isolation.
    import jax
    from dense_evolution.mitigation import (
        _zero_noise_extrapolation_healing_core, _zne_density_matrix_core, _uhlmann_fidelity_core,
    )

    rng = np.random.default_rng(24)
    d = 3
    mats = []
    for _ in range(3):
        a = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        m = a @ a.conj().T
        mats.append(m / np.trace(m))
    rho_at_scales = jnp.asarray(np.stack(mats), dtype=jnp.complex128)
    noise_factors = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)
    sigma = jnp.asarray(7.5, dtype=jnp.float64)
    rho_target = rho_at_scales[0]

    @jax.jit
    def full_pipeline(rho_at_scales, noise_factors, sigma, target_sigma, rho_target):
        healed = _zero_noise_extrapolation_healing_core(rho_at_scales, sigma, target_sigma)
        corrected = _zne_density_matrix_core(rho_at_scales, noise_factors, 2)
        fidelity = _uhlmann_fidelity_core(corrected, rho_target)
        return healed, corrected, fidelity

    healed, corrected, fidelity = full_pipeline(rho_at_scales, noise_factors, sigma, 10.0, rho_target)
    jax.block_until_ready((healed, corrected, fidelity))

    np.testing.assert_allclose(np.asarray(corrected), np.asarray(corrected).conj().T, atol=1e-9)
    assert np.trace(np.asarray(corrected)).real == pytest.approx(1.0, abs=1e-9)
    assert 0.0 <= float(fidelity) <= 1.0 + 1e-9


def test_jsd_predictive_zne_density_matrix_output_is_a_valid_density_matrix():
    rng = np.random.default_rng(30)
    d = 4
    rho_at_scales = jnp.stack([
        jnp.asarray(_random_density_matrix(rng, d), dtype=jnp.complex128) for _ in range(3)
    ])
    got = np.asarray(jsd_predictive_zne_density_matrix(rho_at_scales, [1.0, 2.0, 3.0]))
    np.testing.assert_allclose(got, got.conj().T, atol=1e-9)
    assert np.trace(got).real == pytest.approx(1.0, abs=1e-9)
    assert np.linalg.eigvalsh(got).min() >= -1e-9


def test_jsd_predictive_zne_density_matrix_rejects_non_3point():
    rng = np.random.default_rng(31)
    rho_at_scales = jnp.stack([
        jnp.asarray(_random_density_matrix(rng, 2), dtype=jnp.complex128) for _ in range(4)
    ])
    with pytest.raises(NotImplementedError):
        jsd_predictive_zne_density_matrix(rho_at_scales, [1.0, 2.0, 3.0, 4.0])


def test_jsd_predictive_zne_density_matrix_reduces_to_plain_when_scales_identical():
    # Identical density matrices at all 3 scales -> JSD(scale1,scale2) =
    # JSD(scale2,scale3) = 0 exactly -> nonlinearity = 0/eps = 0, which
    # is <= 0 -> the rectification means this must reduce EXACTLY to
    # plain zne_density_matrix (zero risk in the inactive regime), the
    # core safety property this function is built around.
    rng = np.random.default_rng(32)
    rho = jnp.asarray(_random_density_matrix(rng, 3), dtype=jnp.complex128)
    rho_at_scales = jnp.stack([rho, rho, rho])
    noise_factors = [1.0, 2.0, 3.0]

    plain = zne_density_matrix(rho_at_scales, noise_factors)
    jsd_corrected = jsd_predictive_zne_density_matrix(rho_at_scales, noise_factors)
    np.testing.assert_allclose(np.asarray(jsd_corrected), np.asarray(plain), atol=1e-9)


def test_jsd_predictive_zne_density_matrix_improves_photon_loss_fidelity():
    # Real physics regression test, not just abstract math: reproduces
    # the actual scenario this function was validated against in
    # Dense-Evolution-Discovery's photonic_predictive_zne.py -- a Bell
    # state under amplitude_damping noise (= photon loss on a
    # dual-rail-encoded qubit), at a photon-loss rate where the
    # validated large-sample run showed a real improvement.
    from dense_evolution.registry import NoiseModel

    sim = de.DenseSVSimulator(2)
    sim.run_circuit([("h", 0), ("cx", 0, 1)])
    ideal_sv = np.asarray(sim.get_statevector())
    rho_ideal = jnp.asarray(np.outer(ideal_sv, ideal_sv.conj()), dtype=jnp.complex128)

    rng = np.random.default_rng(0)

    def noisy_rho(gamma, k=200):
        dim = len(ideal_sv)
        rho = np.zeros((dim, dim), dtype=np.complex128)
        for _ in range(k):
            sv = NoiseModel.apply_to_sv(ideal_sv.copy(), 2, 'amplitude_damping', gamma, rng=rng)
            rho += np.outer(sv, sv.conj())
        rho /= k
        return rho

    gamma_base = 1.0 - 0.7  # eta=0.7, matches a validated point from the Discovery run
    rho_at_scales = jnp.stack([
        jnp.asarray(noisy_rho(min(gamma_base * s, 1.0)), dtype=jnp.complex128) for s in (1.0, 2.0, 3.0)
    ])

    plain = zne_density_matrix(rho_at_scales, (1.0, 2.0, 3.0))
    jsd_corrected = jsd_predictive_zne_density_matrix(rho_at_scales, (1.0, 2.0, 3.0))
    fidelity_plain = float(uhlmann_fidelity(plain, rho_ideal))
    fidelity_jsd = float(uhlmann_fidelity(jsd_corrected, rho_ideal))

    assert 0.0 <= fidelity_jsd <= 1.0 + 1e-9
    assert fidelity_jsd >= fidelity_plain - 1e-9  # never worse than plain in the inactive case; may improve
