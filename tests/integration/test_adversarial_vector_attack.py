"""
Tests for ia_utils.adversarial_vector_attack -- the gradient-based
Phi-Trigger stress test (arXiv:2607.27465-inspired) for
enhanced_dense_healing_hybrid. Verifies both the isolated
craft_adversarial_healing_perturbation call AND the crafted
perturbation's effect on the real end-to-end healing pipeline, not
just the isolated trigger-magnitude proxy.
"""
import numpy as np
import pytest

from ia_utils.adversarial_vector_attack import craft_adversarial_healing_perturbation
from ia_utils.vector_healing import enhanced_dense_healing_hybrid


def _static_segment(seed, n=30, dim=8):
    """A near-constant sequence -- the Phi-Trigger should classify this
    as static (median-replaced) almost everywhere."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=dim) * 0.3
    return np.tile(base, (n, 1)) + rng.normal(size=(n, dim)) * 0.001


def _dynamic_segment(seed, n=30, dim=8):
    """Ordinary iid noise at a scale that keeps calculate_phi_ab's
    distance term within MAX_SEMANTIC_DISTANCE (avoids the clip-
    saturation edge case verified separately below) -- the Phi-Trigger
    should classify this as dynamic (kept as-is) almost everywhere."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, dim)) * 0.3


class TestInputValidation:

    def test_rejects_target_idx_below_2(self):
        vettori = _dynamic_segment(0)
        with pytest.raises(ValueError, match="target_idx"):
            craft_adversarial_healing_perturbation(vettori, 1)

    def test_rejects_target_idx_at_or_beyond_length(self):
        vettori = _dynamic_segment(0)
        with pytest.raises(ValueError, match="target_idx"):
            craft_adversarial_healing_perturbation(vettori, len(vettori))

    def test_rejects_unknown_direction(self):
        vettori = _dynamic_segment(0)
        with pytest.raises(ValueError, match="direction"):
            craft_adversarial_healing_perturbation(vettori, 15, direction="sideways")

    def test_rejects_non_positive_epsilon(self):
        vettori = _dynamic_segment(0)
        with pytest.raises(ValueError, match="epsilon"):
            craft_adversarial_healing_perturbation(vettori, 15, epsilon=0.0)


class TestEpsilonBudget:

    def test_perturbation_norm_never_exceeds_epsilon(self):
        vettori = _dynamic_segment(0)
        for epsilon in (0.05, 0.2, 1.0, 3.0):
            result = craft_adversarial_healing_perturbation(
                vettori, 15, epsilon=epsilon, direction="flip_to_static")
            assert result["perturbation_norm"] <= epsilon + 1e-9

    def test_larger_epsilon_is_never_worse_than_smaller(self):
        # BUG FIX regression: step_size used to scale with epsilon
        # (2*epsilon/n_steps), which made a LARGER budget overshoot and
        # converge to a WORSE (higher, for flip_to_static) final
        # magnitude than a smaller budget -- the opposite of what any
        # correct optimizer should do (more room to move should never
        # hurt). Verified directly: final_magnitude must be
        # non-increasing as epsilon grows, for a fixed seed/target.
        vettori = _dynamic_segment(0)
        epsilons = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0]
        magnitudes = []
        for epsilon in epsilons:
            result = craft_adversarial_healing_perturbation(
                vettori, 15, epsilon=epsilon, n_steps=50, direction="flip_to_static")
            magnitudes.append(result["final_magnitude"])
        for earlier, later in zip(magnitudes, magnitudes[1:]):
            assert later <= earlier + 1e-9, (
                f"larger epsilon gave a worse (higher) final magnitude: {magnitudes}")


class TestFlipToStatic:
    """Suppression direction: push an originally-dynamic (kept-as-is)
    point across the threshold so the trigger discards it as noise."""

    def test_flips_the_isolated_trigger_decision(self):
        vettori = _dynamic_segment(0)
        result = craft_adversarial_healing_perturbation(
            vettori, 15, epsilon=1.0, n_steps=50, direction="flip_to_static")
        assert result["original_trigger_active"] is True
        assert result["success"] is True
        assert result["final_trigger_active"] is False
        assert result["final_magnitude"] < result["original_magnitude"]

    def test_flips_the_real_end_to_end_healing_decision(self):
        # The point of this utility: the crafted perturbation must
        # change what the REAL enhanced_dense_healing_hybrid pipeline
        # does, not just an isolated proxy computation.
        vettori = _dynamic_segment(0)
        target_idx = 15
        result = craft_adversarial_healing_perturbation(
            vettori, target_idx, epsilon=1.0, n_steps=50, direction="flip_to_static")
        assert result["success"] is True

        healed_orig, _ = enhanced_dense_healing_hybrid(vettori.copy())
        healed_pert, _ = enhanced_dense_healing_hybrid(result["perturbed_vettori"].copy())

        # Originally kept as-is (dynamic): healed output equals raw input.
        assert np.allclose(healed_orig[target_idx], vettori[target_idx])
        # After the attack: median-replaced instead, so it must differ
        # from the (perturbed) input that was fed in.
        assert not np.allclose(healed_pert[target_idx], result["perturbed_vettori"][target_idx])


class TestFlipToDynamic:
    """Evasion direction: push an originally-static (median-replaced)
    point across the threshold so the trigger keeps it as-is instead."""

    def test_flips_the_isolated_trigger_decision(self):
        vettori = _static_segment(1)
        result = craft_adversarial_healing_perturbation(
            vettori, 15, epsilon=1.0, n_steps=50, direction="flip_to_dynamic")
        assert result["original_trigger_active"] is False
        assert result["success"] is True
        assert result["final_trigger_active"] is True
        assert result["final_magnitude"] > result["original_magnitude"]

    def test_flips_the_real_end_to_end_healing_decision(self):
        vettori = _static_segment(1)
        target_idx = 15
        result = craft_adversarial_healing_perturbation(
            vettori, target_idx, epsilon=1.0, n_steps=50, direction="flip_to_dynamic")
        assert result["success"] is True

        healed_orig, _ = enhanced_dense_healing_hybrid(vettori.copy())
        healed_pert, _ = enhanced_dense_healing_hybrid(result["perturbed_vettori"].copy())

        # Originally median-replaced (static): healed output must NOT
        # equal the raw (noisy) input at that index.
        assert not np.allclose(healed_orig[target_idx], vettori[target_idx])
        # After the attack: evades the healer, passes through unhealed,
        # so the healed output must equal the (perturbed) input exactly.
        assert np.allclose(healed_pert[target_idx], result["perturbed_vettori"][target_idx])


class TestGradientSaturationEdgeCase:
    """calculate_phi_ab clips to [0, 1] -- when the clip saturates (the
    unclipped weighted sum falls outside that range), the local
    gradient is exactly zero, so a gradient-based attack cannot move at
    all from that starting point. This is a real property of the
    underlying formula (see dense_evolution.healing), not a bug in the
    attack -- verified directly that it's detectable via
    perturbation_norm == 0 with success == False, not silently
    misreported as a "found nothing better" result."""

    def test_saturated_phi_ab_gives_zero_perturbation_and_no_success(self):
        rng = np.random.default_rng(0)
        n, dim = 30, 8
        # Unscaled noise: distance_A_B regularly exceeds
        # MAX_SEMANTIC_DISTANCE=sqrt(2), saturating phi_ab's clip to 0.
        vettori = rng.normal(size=(n, dim))
        result = craft_adversarial_healing_perturbation(
            vettori, 15, epsilon=0.15, n_steps=30, direction="flip_to_dynamic")
        assert result["perturbation_norm"] == 0.0
        assert result["success"] is False
