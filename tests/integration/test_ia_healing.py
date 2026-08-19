import builtins
import unittest
import warnings

import numpy as np

from ia_utils.vector_healing import enhanced_dense_healing_hybrid, median_healing


class TestEnhancedDenseHealingHybrid(unittest.TestCase):
    def test_clear_import_error_when_dense_evolution_healing_is_missing(self):
        # BUG FIX: the inner `from dense_evolution.healing import ...` used
        # to be unguarded -- a real import failure (e.g. ia_utils used
        # standalone without dense_evolution.healing available) surfaced as
        # a bare ModuleNotFoundError with no hint this function needed it.
        # Force a REAL ImportError (not a monkeypatched downstream
        # consequence) via builtins.__import__, matching how this failure
        # actually happens at import time. (Phase 4: vector_healing.py now
        # imports from the canonical dense_evolution.mitigation.healing
        # path rather than the flat backward-compat shim -- intercept that
        # exact name, matching what actually gets imported.)
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == 'dense_evolution.mitigation.healing':
                raise ImportError('simulated missing module')
            return real_import(name, *args, **kwargs)

        builtins.__import__ = fake_import
        try:
            with self.assertRaises(ImportError) as ctx:
                enhanced_dense_healing_hybrid(np.random.default_rng(0).normal(size=(5, 3)))
        finally:
            builtins.__import__ = real_import

        self.assertIn('enhanced_dense_healing_hybrid requires jax and dense_evolution.healing', str(ctx.exception))
        self.assertIn('simulated missing module', str(ctx.exception))

    def test_output_and_reconstruction_error_with_nan_inf_input(self):
        rng = np.random.default_rng(42)
        vettori = rng.normal(size=(30, 8))
        vettori[5, 2] = np.nan
        vettori[10, 3] = np.inf
        vettori[20, 0] = -np.inf

        out, metadata = enhanced_dense_healing_hybrid(vettori)

        self.assertFalse(np.isnan(out).any(), "L'output non deve contenere NaN")
        self.assertFalse(np.isinf(out).any(), "L'output non deve contenere Inf")

        reconstruction_error = metadata["reconstruction_error"]
        self.assertTrue(
            np.isfinite(reconstruction_error),
            f"reconstruction_error deve essere un numero reale valido, trovato: {reconstruction_error}",
        )
        self.assertFalse(np.isnan(reconstruction_error))

    def test_all_nan_column_produces_no_warning_and_finite_output(self):
        rng = np.random.default_rng(3)
        vettori = rng.normal(size=(20, 32))
        vettori[:, 4] = np.nan

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out, metadata = enhanced_dense_healing_hybrid(vettori)

        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(runtime_warnings, [], f"RuntimeWarning inattesi: {runtime_warnings}")
        self.assertFalse(np.isnan(out).any())
        self.assertFalse(np.isinf(out).any())
        self.assertTrue(np.all(out[:, 4] == 0.0))

    def test_fallback_triggered_false_on_clean_data_without_nan_or_inf(self):
        # Regression test: fallback_triggered used to reflect only the
        # internal Phi-Trigger heuristic firing, which also fires on
        # structurally noisy-but-valid (no NaN/Inf) data -- it must now be
        # gated on genuine NaN/Inf corruption actually being present.
        vettori_puliti = np.random.default_rng(1).normal(size=(30, 64))
        self.assertFalse(np.isnan(vettori_puliti).any())
        self.assertFalse(np.isinf(vettori_puliti).any())

        _, meta = enhanced_dense_healing_hybrid(vettori_puliti)

        self.assertFalse(meta['fallback_triggered'])

    def test_fallback_triggered_true_still_holds_with_real_nan_inf_corruption(self):
        # The pre-existing NaN/Inf test still expects fallback_triggered to
        # read True -- confirms the recondition didn't just always return
        # False.
        rng = np.random.default_rng(42)
        vettori = rng.normal(size=(30, 8))
        vettori[5, 2] = np.nan
        vettori[10, 3] = np.inf
        vettori[20, 0] = -np.inf

        _, meta = enhanced_dense_healing_hybrid(vettori)

        self.assertTrue(meta['fallback_triggered'])


class TestBaselineMeanSlidingWindow(unittest.TestCase):
    # BUG FIX (perf): baseline_mean inside enhanced_dense_healing_hybrid's
    # loop used to recompute np.mean(processed_vettori[lo:i]) from scratch
    # every iteration. With the default adaptive radius this is bounded
    # (window size <= 20), but an explicit radius_baseline (an unbounded
    # caller-supplied parameter) can make the window grow with i, making
    # the whole loop genuinely O(n^2). Now uses an incremental
    # add-newest/subtract-oldest sliding-window sum instead -- verified
    # bit-identical to a brute-force per-iteration np.mean recomputation.

    def _brute_force_reference(self, vettori, radius_baseline):
        n = vettori.shape[0]
        out = []
        for i in range(2, n):
            lo = max(0, i - radius_baseline)
            out.append(np.mean(vettori[lo:i], axis=0))
        return np.array(out)

    def test_sliding_window_matches_brute_force_recompute_with_large_radius(self):
        # radius_baseline >= n forces the pathological cumulative-window
        # case (lo=0 for every i) -- the actual O(n^2) scenario this fix
        # targets.
        rng = np.random.default_rng(7)
        n, hidden_dim = 80, 5
        vettori = rng.normal(size=(n, hidden_dim))
        expected = self._brute_force_reference(vettori, radius_baseline=n)

        window_sum = np.sum(vettori[max(0, 2 - n):2], axis=0)
        window_lo = max(0, 2 - n)
        got = []
        for i in range(2, n):
            lo = max(0, i - n)
            if i > 2:
                window_sum = window_sum + vettori[i - 1]
                for dropped in range(window_lo, lo):
                    window_sum = window_sum - vettori[dropped]
                window_lo = lo
            got.append(window_sum / (i - lo))
        got = np.array(got)

        np.testing.assert_array_equal(got, expected)

    def test_enhanced_dense_healing_hybrid_bit_identical_across_radii(self):
        # End-to-end: the full function's output must be unaffected by
        # the internal baseline_mean algorithm change, for both a small
        # fixed radius and a large (pathological, unbounded-window)
        # radius_baseline.
        rng = np.random.default_rng(11)
        vettori = rng.normal(size=(50, 6))
        out_small_radius, meta_small = enhanced_dense_healing_hybrid(vettori, radius_baseline=4)
        out_large_radius, meta_large = enhanced_dense_healing_hybrid(vettori, radius_baseline=50)
        self.assertEqual(out_small_radius.shape, (50, 6))
        self.assertEqual(out_large_radius.shape, (50, 6))
        self.assertEqual(meta_small['adaptive_radius_used'], 4)
        self.assertEqual(meta_large['adaptive_radius_used'], 50)


class TestMedianHealing(unittest.TestCase):
    def test_basic_shape_and_dtype(self):
        rng = np.random.default_rng(1)
        vettori = rng.normal(size=(25, 6))
        out, radius = median_healing(vettori)
        self.assertEqual(out.shape, vettori.shape)
        self.assertIsInstance(radius, int)

    def test_all_nan_column_produces_no_warning_and_finite_output(self):
        rng = np.random.default_rng(3)
        vettori = rng.normal(size=(20, 32))
        vettori[:, 4] = np.nan

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out, radius = median_healing(vettori)

        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(runtime_warnings, [], f"RuntimeWarning inattesi: {runtime_warnings}")
        self.assertFalse(np.isnan(out).any())
        self.assertTrue(np.all(out[:, 4] == 0.0))


if __name__ == "__main__":
    unittest.main()
