"""
Unit tests for dashboard_core.vector_healing_demo.generate_corrupted_sequence
-- the public duplication of ui_pages/vector_healing.py's private
`_generate_corrupted_sequence`, added so launch_interactive_panel's
ipywidgets Vector Healing panel doesn't depend on the repo-only ui_pages
package (see the module's own docstring).
"""

import numpy as np

from dashboard_core.vector_healing_demo import generate_corrupted_sequence


def test_shapes_and_ideal_is_clean():
    rng = np.random.default_rng(0)
    ideal, corrupted = generate_corrupted_sequence(80, 6, 10, rng)
    assert ideal.shape == (80, 6)
    assert corrupted.shape == (80, 6)
    assert np.all(np.isfinite(ideal))  # the "ideal" reference is never corrupted


def test_corrupted_has_finite_and_nonfinite_values():
    rng = np.random.default_rng(1)
    _, corrupted = generate_corrupted_sequence(80, 6, 20, rng)
    # 20% anomaly rate on 80 steps must introduce at least one non-finite
    # or extreme-spike value -- the whole point of the "corrupted" copy.
    assert not np.all(np.isfinite(corrupted)) or np.any(np.abs(corrupted) > 2.5)


def test_anomaly_count_matches_percentage():
    rng = np.random.default_rng(2)
    n_steps, anomaly_pct = 100, 15
    ideal, corrupted = generate_corrupted_sequence(n_steps, 6, anomaly_pct, rng)
    diff_rows = np.any(~np.isclose(ideal, corrupted, equal_nan=False) | ~np.isfinite(corrupted), axis=1)
    assert diff_rows.sum() >= int(round(n_steps * anomaly_pct / 100.0))


def test_deterministic_given_same_rng_state():
    a_ideal, a_corrupted = generate_corrupted_sequence(50, 4, 10, np.random.default_rng(42))
    b_ideal, b_corrupted = generate_corrupted_sequence(50, 4, 10, np.random.default_rng(42))
    np.testing.assert_array_equal(a_ideal, b_ideal)
    np.testing.assert_array_equal(a_corrupted, b_corrupted)


def test_minimum_one_anomaly_even_at_low_percentage():
    rng = np.random.default_rng(3)
    ideal, corrupted = generate_corrupted_sequence(20, 6, 1, rng)  # rounds to 0 -> floored to 1
    assert not np.array_equal(ideal, corrupted)
