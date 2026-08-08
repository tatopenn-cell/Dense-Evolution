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
        # actually happens at import time.
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == 'dense_evolution.healing':
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
