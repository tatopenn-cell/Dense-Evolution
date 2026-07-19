import unittest

import numpy as np

from ia_utils.vector_healing import enhanced_dense_healing_hybrid


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


if __name__ == "__main__":
    unittest.main()
