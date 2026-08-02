"""
Tests for dashboard_core.mitigation -- real Zero-Noise Extrapolation
(scalar Pauli expectation and density-matrix) wired to dense_evolution's
own real Kraus noise channels, not mocks.
"""

import numpy as np
import pytest

from dashboard_core.mitigation import run_zne_mitigation, run_density_matrix_zne

BELL_QASM = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
    'qreg q[2];\ncreg c[2];\n'
    'h q[0];\ncx q[0],q[1];\n'
    'measure q -> c;\n'
)


class TestScalarZne:

    def test_ideal_expectation_matches_known_bell_zz_value(self):
        # <ZZ> on a Bell state is exactly +1 (perfectly correlated in Z).
        result = run_zne_mitigation(BELL_QASM, "ZZ", "depolarizing", 0.0, seed=1, n_trials=5)
        assert result.ideal_expectation == pytest.approx(1.0, abs=1e-9)

    def test_zero_noise_probability_zne_matches_ideal(self):
        result = run_zne_mitigation(BELL_QASM, "ZZ", "depolarizing", 0.0, seed=1, n_trials=5)
        assert result.zne_extrapolated == pytest.approx(result.ideal_expectation, abs=1e-9)

    def test_default_method_is_richardson_with_3_scales(self):
        result = run_zne_mitigation(BELL_QASM, "ZZ", "depolarizing", 0.05, seed=1, n_trials=50)
        assert result.extrapolation_method == "richardson"
        assert result.noise_factors == [1.0, 2.0, 3.0]

    def test_polynomial_method_uses_5_scales(self):
        result = run_zne_mitigation(
            BELL_QASM, "ZZ", "depolarizing", 0.05, seed=1, n_trials=50, extrapolation_method="polynomial",
        )
        assert result.extrapolation_method == "polynomial"
        assert result.noise_factors == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_polynomial_and_richardson_both_recover_real_noise_close_to_ideal(self):
        # Real decay, not fabricated: both methods should land near the
        # true ideal value for a modest noise_p, each via its own real
        # dense_evolution extrapolation function.
        richardson = run_zne_mitigation(BELL_QASM, "ZZ", "depolarizing", 0.05, seed=3, n_trials=300)
        polynomial = run_zne_mitigation(
            BELL_QASM, "ZZ", "depolarizing", 0.05, seed=3, n_trials=300, extrapolation_method="polynomial",
        )
        assert abs(richardson.zne_extrapolated - richardson.ideal_expectation) < 0.05
        assert abs(polynomial.zne_extrapolated - polynomial.ideal_expectation) < 0.05

    def test_explicit_noise_factors_override_the_method_default(self):
        result = run_zne_mitigation(
            BELL_QASM, "ZZ", "depolarizing", 0.05, seed=1, n_trials=5,
            extrapolation_method="polynomial", noise_factors=(1.0, 2.0, 3.0),
        )
        assert result.noise_factors == [1.0, 2.0, 3.0]

    def test_same_seed_is_deterministic(self):
        r1 = run_zne_mitigation(BELL_QASM, "ZZ", "depolarizing", 0.1, seed=7, n_trials=20)
        r2 = run_zne_mitigation(BELL_QASM, "ZZ", "depolarizing", 0.1, seed=7, n_trials=20)
        assert r1.zne_extrapolated == pytest.approx(r2.zne_extrapolated, abs=1e-9)

    def test_unknown_extrapolation_method_raises(self):
        with pytest.raises(ValueError, match="extrapolation_method"):
            run_zne_mitigation(BELL_QASM, "ZZ", "depolarizing", 0.05, extrapolation_method="quadratic")

    def test_unknown_noise_model_raises(self):
        with pytest.raises(ValueError, match="unknown noise model"):
            run_zne_mitigation(BELL_QASM, "ZZ", "not_a_real_channel", 0.05)

    def test_wrong_pauli_string_length_raises(self):
        with pytest.raises(ValueError, match="pauli_string length"):
            run_zne_mitigation(BELL_QASM, "ZZZ", "depolarizing", 0.05)


class TestDensityMatrixZne:

    def test_zero_noise_probability_fidelity_is_one(self):
        result = run_density_matrix_zne(BELL_QASM, "depolarizing", 0.0, seed=1, n_trials=5)
        assert result.fidelity_raw == pytest.approx(1.0, abs=1e-6)
        assert result.fidelity_corrected == pytest.approx(1.0, abs=1e-6)

    def test_correction_improves_or_matches_fidelity(self):
        result = run_density_matrix_zne(BELL_QASM, "depolarizing", 0.05, seed=1, n_trials=200)
        assert result.fidelity_corrected >= result.fidelity_raw - 1e-6

    def test_fidelities_are_valid_probabilities(self):
        result = run_density_matrix_zne(BELL_QASM, "depolarizing", 0.2, seed=2, n_trials=100)
        assert 0.0 <= result.fidelity_raw <= 1.0 + 1e-9
        assert 0.0 <= result.fidelity_corrected <= 1.0 + 1e-9

    def test_unknown_noise_model_raises(self):
        with pytest.raises(ValueError, match="unknown noise model"):
            run_density_matrix_zne(BELL_QASM, "not_a_real_channel", 0.05)
