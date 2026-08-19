"""
Unit tests for dense_evolution/mitigation/renyi.py -- sandwiched quantum
Renyi divergence. Cross-checked against Dense-Evolution-Discovery
Experiment 29
(https://tatopenn-cell.github.io/Dense-Evolution-Discovery/sandwiched_renyi_density_matrix/),
where the same fixed formula was validated against the relative-entropy
limit, the classical/diagonal reduction, and hand-derived support-mismatch
cases.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_evolution.mitigation import (
    sandwiched_renyi_divergence, sandwiched_renyi_divergence_jit, uhlmann_fidelity,
)


def _depolarize(rho, p):
    d = rho.shape[0]
    return (1 - p) * rho + p * jnp.eye(d, dtype=jnp.complex128) / d


def _bell_state_rho():
    sv = jnp.array([1.0, 0.0, 0.0, 1.0], dtype=jnp.complex128) / jnp.sqrt(2.0)
    return jnp.outer(sv, jnp.conj(sv))


def _amplitude_damping_2q(rho, p):
    k0 = jnp.array([[1.0, 0.0], [0.0, jnp.sqrt(1.0 - p)]], dtype=jnp.complex128)
    k1 = jnp.array([[0.0, jnp.sqrt(p)], [0.0, 0.0]], dtype=jnp.complex128)
    identity_1q = jnp.eye(2, dtype=jnp.complex128)
    K0 = jnp.kron(k0, identity_1q)
    K1 = jnp.kron(k1, identity_1q)
    return (K0 @ rho @ jnp.conj(K0).T) + (K1 @ rho @ jnp.conj(K1).T)


def _classical_renyi_divergence(p, q, alpha):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    term = np.sum(p ** alpha * q ** (1.0 - alpha))
    return (1.0 / (alpha - 1.0)) * np.log2(term)


def test_identical_states_give_zero_at_every_alpha():
    rho = _amplitude_damping_2q(_bell_state_rho(), 0.3)
    for alpha in (0.5, 0.8, 1.0, 1.5, 2.0):
        d = sandwiched_renyi_divergence(rho, rho, alpha=alpha)
        assert abs(d) < 1e-6, f"D_alpha(rho||rho) should be 0 at alpha={alpha}, got {d}"


def test_diagonal_case_matches_classical_renyi_formula():
    p = np.array([0.6, 0.3, 0.1])
    q = np.array([0.5, 0.3, 0.2])
    rho = jnp.array(np.diag(p), dtype=jnp.complex128)
    sigma = jnp.array(np.diag(q), dtype=jnp.complex128)
    for alpha in (0.6, 1.8, 2.5):
        d_quantum = sandwiched_renyi_divergence(rho, sigma, alpha=alpha)
        d_classical = _classical_renyi_divergence(p, q, alpha)
        assert d_quantum == pytest.approx(d_classical, abs=1e-6)


def test_support_violation_gives_infinity_for_alpha_greater_than_1():
    sv1 = jnp.array([1.0, 0.0], dtype=jnp.complex128)
    sv2 = jnp.array([np.cos(0.7), np.sin(0.7)], dtype=jnp.complex128)
    rho = jnp.outer(sv1, jnp.conj(sv1))
    sigma = jnp.outer(sv2, jnp.conj(sv2))
    d = sandwiched_renyi_divergence(rho, sigma, alpha=1.5)
    assert np.isinf(d) and d > 0


def test_no_support_violation_below_alpha_1():
    sv1 = jnp.array([1.0, 0.0], dtype=jnp.complex128)
    sv2 = jnp.array([np.cos(0.7), np.sin(0.7)], dtype=jnp.complex128)
    rho = jnp.outer(sv1, jnp.conj(sv1))
    sigma = jnp.outer(sv2, jnp.conj(sv2))
    d = sandwiched_renyi_divergence(rho, sigma, alpha=0.7)
    assert np.isfinite(d)


def test_alpha_one_limit_matches_relative_entropy_on_full_rank_states():
    rho = _amplitude_damping_2q(_bell_state_rho(), 0.1)
    sigma = _amplitude_damping_2q(_bell_state_rho(), 0.4)
    rho = _depolarize(rho, 0.1)
    sigma = _depolarize(sigma, 0.1)

    ev_r, ec_r = np.linalg.eigh(np.array(rho))
    ev_s, ec_s = np.linalg.eigh(np.array(sigma))
    ev_r = np.clip(ev_r, 1e-14, None)
    ev_s = np.clip(ev_s, 1e-14, None)
    log_rho = ec_r @ np.diag(np.log2(ev_r)) @ ec_r.conj().T
    log_sigma = ec_s @ np.diag(np.log2(ev_s)) @ ec_s.conj().T
    d_ref = float(np.real(np.trace(np.array(rho) @ (log_rho - log_sigma))))

    d_case_one = sandwiched_renyi_divergence(rho, sigma, alpha=1.0)
    assert d_case_one == pytest.approx(d_ref, abs=1e-3)


def test_jit_matches_eager():
    rho = _amplitude_damping_2q(_bell_state_rho(), 0.2)
    sigma = _amplitude_damping_2q(_bell_state_rho(), 0.5)
    for alpha in (0.5, 1.0, 1.5, 2.0):
        eager = sandwiched_renyi_divergence(rho, sigma, alpha=alpha)
        jitted = float(sandwiched_renyi_divergence_jit(rho, sigma, alpha))
        if np.isinf(eager):
            assert np.isinf(jitted)
        else:
            assert eager == pytest.approx(jitted, abs=1e-9)


def test_no_nan_across_noise_sweep():
    rho = _bell_state_rho()
    for p in (0.0, 0.2, 0.5, 0.8, 0.99):
        rho_noisy = _amplitude_damping_2q(rho, p)
        for alpha in (0.5, 1.0, 1.5, 2.0):
            d = sandwiched_renyi_divergence(rho, rho_noisy, alpha=alpha)
            assert not np.isnan(d)


def test_differentiable_through_both_arguments():
    def loss(theta):
        sv1 = jnp.array([1.0, 0.0], dtype=jnp.complex128)
        rho = jnp.outer(sv1, jnp.conj(sv1))
        sv2 = jnp.array([jnp.cos(theta), jnp.sin(theta)], dtype=jnp.complex128)
        sigma = jnp.outer(sv2, jnp.conj(sv2))
        sigma = 0.9 * sigma + 0.1 * jnp.eye(2, dtype=jnp.complex128) / 2.0
        rho = 0.9 * rho + 0.1 * jnp.eye(2, dtype=jnp.complex128) / 2.0
        from dense_evolution.mitigation.renyi import _sandwiched_renyi_divergence_core
        return _sandwiched_renyi_divergence_core(rho, sigma, 1.5).real

    g = jax.grad(loss)(0.3)
    assert not jnp.isnan(g)


def test_qualitatively_distinct_from_uhlmann_fidelity_on_a_noise_sweep():
    # Not asserting strict monotonicity for the divergence -- Experiment 29
    # found and documented a real non-monotonic dip in this exact sweep
    # shape, honestly reported rather than smoothed over (no theorem in
    # Muller-Lennert et al. requires monotonicity along an arbitrary
    # one-parameter noise sweep). Just checks the two metrics are not a
    # trivial rescaling of one another -- the actual motivation for
    # promoting the divergence alongside fidelity (Experiment 29, Part 4).
    rho = _bell_state_rho()
    fids, divs = [], []
    for p in (0.0, 0.2, 0.4, 0.6, 0.8):
        rho_noisy = _amplitude_damping_2q(rho, p)
        fids.append(uhlmann_fidelity(rho, rho_noisy))
        divs.append(sandwiched_renyi_divergence(rho, rho_noisy, alpha=1.5))
    assert fids == sorted(fids, reverse=True)  # fidelity decreases monotonically
    assert divs[-1] > divs[0]  # divergence net-increases from p=0 to p=0.8
    assert divs[-1] / max(divs[1], 1e-9) != pytest.approx(fids[-1] / max(fids[1], 1e-9), rel=0.05)
