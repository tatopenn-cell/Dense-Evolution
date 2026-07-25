import unittest
import warnings

import numpy as np

from ia_utils.vector_healing import enhanced_dense_healing_hybrid, median_healing


class TestEnhancedDenseHealingHybrid(unittest.TestCase):
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
