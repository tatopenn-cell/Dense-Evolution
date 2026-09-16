"""Regression tests for the cold/warm split in
ia_utils.vector_healing.enhanced_dense_healing_hybrid's 'phi' branch
(issue #258 point 7): the first time a given sequence length is seen this
process, it must go through the per-index loop (no XLA compile for a new
vmap batch shape); once that length recurs, the vmapped path takes over.
Both must produce numerically identical output for identical input --
that's the correctness bar the issue itself asks for before shipping."""
import numpy as np

from ia_utils.vector_healing import (
    _mark_and_check_warm,
    _PHI_VMAP_WARM_MAXSIZE,
    _PHI_VMAP_WARM_SIZES,
    enhanced_dense_healing_hybrid,
)


def test_cold_loop_path_matches_warm_vmap_path_exactly():
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((30, 4))
    _PHI_VMAP_WARM_SIZES.clear()

    cold_healed, cold_meta = enhanced_dense_healing_hybrid(vectors.copy(), trigger_mode='phi')
    assert 28 in _PHI_VMAP_WARM_SIZES  # n - 2 batch size now marked warm

    warm_healed, warm_meta = enhanced_dense_healing_hybrid(vectors.copy(), trigger_mode='phi')

    np.testing.assert_allclose(cold_healed, warm_healed)
    assert cold_meta['fallback_triggered'] == warm_meta['fallback_triggered']
    assert cold_meta['adaptive_radius_used'] == warm_meta['adaptive_radius_used']


def test_cold_loop_path_heals_an_injected_spike():
    rng = np.random.default_rng(1)
    vectors = rng.standard_normal((15, 3))
    vectors[8, 0] += 10.0
    _PHI_VMAP_WARM_SIZES.clear()  # force the cold (per-index loop) path

    healed, _ = enhanced_dense_healing_hybrid(vectors, trigger_mode='phi')
    assert abs(healed[8][0]) < 3.0


def test_warm_lru_is_bounded_and_evicts_oldest():
    _PHI_VMAP_WARM_SIZES.clear()
    for n in range(_PHI_VMAP_WARM_MAXSIZE + 3):
        _mark_and_check_warm(n)
    assert len(_PHI_VMAP_WARM_SIZES) == _PHI_VMAP_WARM_MAXSIZE
    assert 0 not in _PHI_VMAP_WARM_SIZES  # oldest entries evicted first
    assert (_PHI_VMAP_WARM_MAXSIZE + 2) in _PHI_VMAP_WARM_SIZES


def test_mark_and_check_warm_returns_false_then_true():
    _PHI_VMAP_WARM_SIZES.clear()
    assert _mark_and_check_warm(42) is False
    assert _mark_and_check_warm(42) is True
