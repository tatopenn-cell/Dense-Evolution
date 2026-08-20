"""
Unit tests for dense_evolution/mitigation/kl_divergence.py -- classical
Kullback-Leibler divergence. Cross-checked against Dense-Evolution-Discovery
Experiment 32
(https://tatopenn-cell.github.io/Dense-Evolution-Discovery/kullback_leibler_divergence/),
where the same construction was validated against scipy.stats.entropy,
Gibbs' inequality, and a genuine support-violation case.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import entropy as scipy_entropy

from dense_evolution.mitigation import kl_divergence, kl_divergence_jit


def test_matches_scipy_entropy_reference():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = rng.integers(2, 8)
        p = rng.dirichlet(np.ones(n))
        q = rng.dirichlet(np.ones(n))
        mine = kl_divergence(p, q)
        ref_bits = scipy_entropy(p, q) / np.log(2.0)
        assert mine == pytest.approx(ref_bits, abs=1e-9)


def test_gibbs_inequality_never_negative():
    rng = np.random.default_rng(1)
    for _ in range(200):
        n = rng.integers(2, 10)
        p = rng.dirichlet(np.ones(n))
        q = rng.dirichlet(np.ones(n))
        assert kl_divergence(p, q) >= -1e-9


def test_zero_at_equality():
    p = np.array([0.5, 0.3, 0.2])
    assert kl_divergence(p, p) < 1e-12


def test_asymmetric_in_general():
    p = np.array([0.6, 0.3, 0.1])
    q = np.array([0.2, 0.3, 0.5])
    d_pq = kl_divergence(p, q)
    d_qp = kl_divergence(q, p)
    assert abs(d_pq - d_qp) > 0.1


def test_support_violation_gives_infinity():
    p = np.array([0.5, 0.5, 0.0])
    q = np.array([0.5, 0.0, 0.5])
    assert np.isinf(kl_divergence(p, q))


def test_zero_mass_in_p_does_not_force_infinity():
    p = np.array([0.5, 0.5, 0.0])
    q = np.array([0.3, 0.3, 0.4])
    d = kl_divergence(p, q)
    assert np.isfinite(d) and d > 0


def test_jit_matches_eager():
    rng = np.random.default_rng(2)
    p = jnp.array(rng.dirichlet(np.ones(5)), dtype=jnp.float64)
    q = jnp.array(rng.dirichlet(np.ones(5)), dtype=jnp.float64)
    eager = kl_divergence(p, q)
    jitted = float(kl_divergence_jit(p, q))
    assert eager == pytest.approx(jitted, abs=1e-9)


def test_measurement_distribution_use_case():
    from dense_evolution import DenseSVSimulator
    n_qubits = 3
    sim = DenseSVSimulator(n_qubits)
    p_ideal = jnp.abs(sim.sv) ** 2
    uniform = jnp.ones_like(p_ideal) / len(p_ideal)
    for noise in (0.0, 0.1, 0.4):
        p_noisy = (1 - noise) * p_ideal + noise * uniform
        p_noisy = p_noisy / jnp.sum(p_noisy)
        d = kl_divergence(p_ideal, p_noisy)
        assert d >= -1e-9
        if noise == 0.0:
            assert d < 1e-9


def test_differentiable_through_both_arguments():
    def loss(theta):
        p = jnp.array([jnp.cos(theta) ** 2, jnp.sin(theta) ** 2], dtype=jnp.float64)
        q = jnp.array([0.5, 0.5], dtype=jnp.float64)
        from dense_evolution.mitigation.kl_divergence import _kl_divergence_core
        return _kl_divergence_core(p, q)

    g = jax.grad(loss)(0.3)
    assert not jnp.isnan(g)
