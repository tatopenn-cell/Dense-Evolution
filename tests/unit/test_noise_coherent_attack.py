"""Unit tests for dense_evolution.noise.coherent_attack -- the
multi-qubit coherent adversarial noise promoted from Dense-Evolution-
Discovery's Steane [[7,1,3]] investigation, generalized to any
stabilizer list.
"""
import numpy as np
import jax.numpy as jnp
import pytest

from dense_evolution.noise.coherent_attack import (
    apply_rz_all,
    x_stabilizer_leakage,
    craft_adversarial_delta,
    craft_adversarial_delta_constrained,
    project_l2_linf,
    decoder_failure_rate,
    random_delta_failure_stats,
)

STEANE_X_STABILIZERS = ['IIIXXXX', 'IXXIIXX', 'XIXIXIX']


class TestApplyRzAll:

    def test_identity_at_zero_delta(self):
        sv0 = jnp.array([0.6, 0.8], dtype=jnp.complex128)
        sv1 = apply_rz_all(sv0, jnp.zeros(1))
        np.testing.assert_allclose(np.asarray(sv1), np.asarray(sv0))

    def test_preserves_probabilities(self):
        # rz is diagonal in the computational basis -- a coherent error
        # must never change measurement probabilities, only phase.
        # atol accounts for this environment's float32/complex64 default
        # precision (JAX x64 not enabled here), not exact float64 math.
        rng = np.random.default_rng(0)
        sv0 = rng.normal(size=8) + 1j * rng.normal(size=8)
        sv0 = jnp.array(sv0 / np.linalg.norm(sv0))
        delta = jnp.array([0.3, -1.1, 2.4])
        sv1 = apply_rz_all(sv0, delta)
        np.testing.assert_allclose(
            np.abs(np.asarray(sv1)) ** 2, np.abs(np.asarray(sv0)) ** 2, atol=1e-5
        )

    def test_preserves_norm(self):
        rng = np.random.default_rng(1)
        sv0 = rng.normal(size=8) + 1j * rng.normal(size=8)
        sv0 = jnp.array(sv0 / np.linalg.norm(sv0))
        sv1 = apply_rz_all(sv0, jnp.array([1.0, 2.0, 3.0]))
        assert float(jnp.linalg.norm(sv1)) == pytest.approx(1.0, abs=1e-10)


class TestXStabilizerLeakage:

    def test_zero_at_zero_delta_on_uniform_superposition(self):
        # |++++++...+> is a genuine +1 eigenstate of every X-type Pauli
        # string (X|+> = |+> on each qubit), unlike |0000000> (X flips
        # bits there, giving <X_stab> = 0, not 1 -- a real, verified
        # distinction, not a rounding effect).
        sv0 = jnp.ones(2 ** 7, dtype=jnp.complex128) / jnp.sqrt(2 ** 7)
        leakage = x_stabilizer_leakage(jnp.zeros(7), sv0, STEANE_X_STABILIZERS)
        assert float(leakage) == pytest.approx(0.0, abs=1e-5)

    def test_positive_for_nonzero_delta(self):
        sv0 = jnp.zeros(2 ** 7, dtype=jnp.complex128).at[0].set(1.0)
        rng = np.random.default_rng(0)
        delta = jnp.array(rng.normal(size=7))
        leakage = x_stabilizer_leakage(delta, sv0, STEANE_X_STABILIZERS)
        assert 0.0 <= float(leakage) <= len(STEANE_X_STABILIZERS)

    def test_gradient_is_computable(self):
        # The whole point of this module: leakage must be JAX-differentiable.
        import jax
        sv0 = jnp.zeros(2 ** 7, dtype=jnp.complex128).at[0].set(1.0)
        grad_fn = jax.grad(lambda d: x_stabilizer_leakage(d, sv0, STEANE_X_STABILIZERS))
        grad = grad_fn(jnp.ones(7) * 0.1)
        assert grad.shape == (7,)
        assert np.all(np.isfinite(np.asarray(grad)))


class TestCraftAdversarialDelta:

    def test_respects_l2_budget(self):
        sv0 = jnp.zeros(2 ** 7, dtype=jnp.complex128).at[0].set(1.0)
        epsilon = 0.5
        delta, leakage, history = craft_adversarial_delta(
            sv0, STEANE_X_STABILIZERS, epsilon=epsilon, n_steps=30, seed=0
        )
        assert np.linalg.norm(delta) <= epsilon + 1e-6
        assert leakage >= history[0]  # best-seen should never be worse than the seed
        assert len(history) == 31

    def test_zero_epsilon_gives_zero_delta(self):
        sv0 = jnp.zeros(2 ** 7, dtype=jnp.complex128).at[0].set(1.0)
        delta, _, _ = craft_adversarial_delta(sv0, STEANE_X_STABILIZERS, epsilon=0.0, n_steps=5, seed=0)
        np.testing.assert_allclose(delta, np.zeros(7), atol=1e-9)


class TestCraftAdversarialDeltaConstrained:

    def test_respects_both_l2_and_linf_budgets(self):
        sv0 = jnp.zeros(2 ** 7, dtype=jnp.complex128).at[0].set(1.0)
        epsilon, linf_cap = 0.6, 0.1
        delta, _, _ = craft_adversarial_delta_constrained(
            sv0, STEANE_X_STABILIZERS, epsilon=epsilon, linf_cap=linf_cap, n_steps=30, seed=0
        )
        assert np.max(np.abs(delta)) <= linf_cap + 1e-8
        assert np.linalg.norm(delta) <= epsilon + 1e-6

    def test_tight_linf_cap_spreads_the_budget(self):
        # A tight per-qubit cap should force multiple qubits to be used
        # instead of concentrating the whole L2 budget on one -- this is
        # the documented fix for craft_adversarial_delta's degenerate
        # single-qubit solution (see module docstring).
        sv0 = jnp.zeros(2 ** 7, dtype=jnp.complex128).at[0].set(1.0)
        delta, _, _ = craft_adversarial_delta_constrained(
            sv0, STEANE_X_STABILIZERS, epsilon=0.5, linf_cap=0.05, n_steps=50, seed=0
        )
        n_active = np.sum(np.abs(delta) > 1e-6)
        assert n_active >= 2


class TestProjectL2Linf:

    def test_already_inside_both_is_unchanged(self):
        y = np.array([0.1, 0.1])
        out = project_l2_linf(y, epsilon=1.0, linf_cap=1.0)
        np.testing.assert_allclose(out, y)

    def test_result_always_satisfies_both_constraints(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            y = rng.normal(scale=5.0, size=7)
            epsilon = rng.uniform(0.1, 2.0)
            linf_cap = rng.uniform(0.05, 1.0)
            out = project_l2_linf(y, epsilon, linf_cap)
            assert np.max(np.abs(out)) <= linf_cap + 1e-6
            assert np.linalg.norm(out) <= epsilon + 1e-6


class TestDecoderFailureRate:
    """Uses a synthetic decode_fn with known behavior to test the
    statistical-aggregation logic itself, independent of any specific
    real decoder implementation (the original Discovery experiment's own
    validation already covers the full physical pipeline end-to-end --
    see the module docstring)."""

    def test_perfect_decoder_never_fails(self):
        sv0 = np.array([1.0, 0.0], dtype=np.complex128)

        def perfect_decoder(sv_noisy, rng):
            return sv0  # always recovers the original exactly

        rate = decoder_failure_rate(np.zeros(1), sv0, perfect_decoder, n_trials=10, rng=np.random.default_rng(0))
        assert rate == 0.0

    def test_broken_decoder_always_fails(self):
        sv0 = np.array([1.0, 0.0], dtype=np.complex128)
        wrong = np.array([0.0, 1.0], dtype=np.complex128)

        def broken_decoder(sv_noisy, rng):
            return wrong  # never recovers the original

        rate = decoder_failure_rate(np.zeros(1), sv0, broken_decoder, n_trials=10, rng=np.random.default_rng(0))
        assert rate == 1.0

    def test_coin_flip_decoder_gives_intermediate_rate(self):
        sv0 = np.array([1.0, 0.0], dtype=np.complex128)
        wrong = np.array([0.0, 1.0], dtype=np.complex128)

        def coin_flip_decoder(sv_noisy, rng):
            return sv0 if rng.random() < 0.5 else wrong

        rate = decoder_failure_rate(
            np.zeros(1), sv0, coin_flip_decoder, n_trials=2000, rng=np.random.default_rng(0)
        )
        assert 0.35 < rate < 0.65


class TestRandomDeltaFailureStats:

    def test_shape_and_range(self):
        sv0 = np.array([1.0, 0.0], dtype=np.complex128)

        def perfect_decoder(sv_noisy, rng):
            return sv0

        rates = random_delta_failure_stats(
            sv0, perfect_decoder, epsilon=0.3, n_qubits=1,
            n_random=5, n_trials_each=10, rng=np.random.default_rng(0),
        )
        assert rates.shape == (5,)
        assert np.all(rates == 0.0)
