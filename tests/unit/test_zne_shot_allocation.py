import numpy as np
import pytest

from dense_evolution.mitigation import zne_shot_allocation


def test_equal_sigmas_follow_richardson_coefficients():
    assert zne_shot_allocation([1.0, 2.0, 3.0], 7000).tolist() == [3000, 3000, 1000]


def test_counts_sum_to_budget_and_are_at_least_one():
    for total in (3, 10, 1001, 9999):
        counts = zne_shot_allocation([1.0, 1.5, 2.0, 3.0], total)
        assert counts.sum() == total
        assert counts.min() >= 1


def test_sigmas_weight_the_allocation():
    assert zne_shot_allocation([1.0, 2.0, 3.0], 9000, [1.0, 0.5, 0.25]).tolist() == [5683, 2842, 475]


def test_variance_reduction_matches_prediction():
    rng = np.random.default_rng(0)
    e = np.array([0.7, 0.5, 0.36])
    eta = np.array([3.0, -3.0, 1.0])
    sigma = np.sqrt(1 - e ** 2)

    def variance(counts):
        est = [eta @ (2 * rng.binomial(counts, (1 + e) / 2) / counts - 1) for _ in range(40000)]
        return np.var(est)

    ratio = variance(np.array([3000, 3000, 3000])) / variance(zne_shot_allocation([1.0, 2.0, 3.0], 9000, sigma))
    predicted = 3 * np.sum(eta ** 2 * sigma ** 2) / np.sum(np.abs(eta) * sigma) ** 2
    assert ratio == pytest.approx(predicted, rel=0.03)


def test_budget_smaller_than_factors_raises():
    with pytest.raises(ValueError, match="at least"):
        zne_shot_allocation([1.0, 2.0, 3.0], 2)
