"""
Unit tests for dense_evolution/mitigation/magic_entropy_shadows.py. Values
and behavior cross-checked against Dense-Evolution-Discovery Experiment 31
(https://tatopenn-cell.github.io/Dense-Evolution-Discovery/quantum_shadows_magic_entropy/),
where the purity-estimator bug fix, median-of-means robustness, and
sample-complexity fit were all independently validated first.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from dense_evolution.mitigation import magic_entropy
from dense_evolution.mitigation.magic_entropy_shadows import (
    approx_shadow_std, fit_shadow_sample_complexity, magic_entropy_from_shadows,
    sample_classical_shadow,
)


def _t_state_rho():
    zero = jnp.array([1.0, 0.0], dtype=jnp.complex128)
    one = jnp.array([0.0, 1.0], dtype=jnp.complex128)
    sv = (zero + jnp.exp(1j * jnp.pi / 4.0) * one) / jnp.sqrt(2.0)
    return jnp.outer(sv, jnp.conj(sv))


def _plus_state_rho():
    zero = jnp.array([1.0, 0.0], dtype=jnp.complex128)
    one = jnp.array([0.0, 1.0], dtype=jnp.complex128)
    sv = (zero + one) / jnp.sqrt(2.0)
    return jnp.outer(sv, jnp.conj(sv))


def test_shadow_snapshots_are_unbiased_on_average():
    rho = _t_state_rho()
    snaps = sample_classical_shadow(rho, 100_000, seed=42)
    empirical_mean = jnp.mean(snaps, axis=0)
    assert float(jnp.max(jnp.abs(empirical_mean - rho))) < 0.03


def test_shadow_snapshots_work_for_a_mixed_state_too():
    # sample_classical_shadow accepts mixed rho, not just pure states --
    # not tested in Discovery's Experiment 31 (which only ever sampled
    # from pure |T>/|+> statevectors).
    mixed = jnp.eye(2, dtype=jnp.complex128) / 2.0
    snaps = sample_classical_shadow(mixed, 100_000, seed=1)
    empirical_mean = jnp.mean(snaps, axis=0)
    assert float(jnp.max(jnp.abs(empirical_mean - mixed))) < 0.03


def test_magic_entropy_from_shadows_converges_for_t_state():
    rho = _t_state_rho()
    m_exact = magic_entropy(rho)
    snaps = sample_classical_shadow(rho, 150_000, seed=3)
    m_hat = magic_entropy_from_shadows(snaps)
    assert abs(m_hat - m_exact) < 0.2


def test_magic_entropy_from_shadows_near_zero_for_stabilizer_state():
    rho = _plus_state_rho()
    snaps = sample_classical_shadow(rho, 150_000, seed=4)
    m_hat = magic_entropy_from_shadows(snaps)
    assert m_hat < 0.25


def test_magic_entropy_from_shadows_rejects_too_few_snapshots():
    rho = _t_state_rho()
    snaps = sample_classical_shadow(rho, 2, seed=0)
    with pytest.raises(ValueError):
        magic_entropy_from_shadows(snaps)


def test_n_groups_is_configurable():
    rho = _t_state_rho()
    snaps = sample_classical_shadow(rho, 60_000, seed=8)
    m_default = magic_entropy_from_shadows(snaps)
    m_5_groups = magic_entropy_from_shadows(snaps, n_groups=5)
    assert abs(m_default) < 2.0  # both just need to run and give a finite bits value
    assert abs(m_5_groups) < 2.0


def test_median_of_means_tolerates_a_corrupted_block():
    from dense_evolution.mitigation.magic_entropy_shadows import _median_of_means
    rng = np.random.default_rng(1)
    values = rng.normal(loc=1.0, scale=0.05, size=2000)
    corrupted = values.copy()
    corrupted[:800] = -1000.0  # 40% corrupted, one contiguous block
    naive_mean = float(np.mean(corrupted))
    mom = _median_of_means(corrupted, n_groups=20)
    assert abs(mom - 1.0) < 1.0
    assert abs(naive_mean - 1.0) > 100.0


def test_approx_shadow_std_decreases_with_more_snapshots():
    assert approx_shadow_std(100_000) < approx_shadow_std(3_000)
    assert approx_shadow_std(100_000) > 0


def test_fit_shadow_sample_complexity_matches_the_module_level_fit_order():
    # Small budget (fast) -- just checks the fitting machinery itself
    # works and lands in a sane range, not a tight reproduction of the
    # full 20-trial study behind approx_shadow_std's built-in constants.
    rho = _t_state_rho()
    m_exact = magic_entropy(rho)
    rows, fit_c, fit_p = fit_shadow_sample_complexity(
        rho, m_exact, n_snapshots_list=(3000, 30000), n_trials=6, seed_base=500,
    )
    assert len(rows) == 2
    assert fit_c > 0
    assert 0.1 < fit_p < 1.0
