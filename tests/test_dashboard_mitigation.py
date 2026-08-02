"""
Tests for dashboard_core.mitigation -- real Zero-Noise Extrapolation
(scalar Pauli-expectation ZNE and density-matrix ZNE), both driven by
dense_evolution's own real noise channels and extrapolation, not a
fabricated mitigated curve. Values below are real, deterministic
outputs (fixed seed) captured directly from the functions, not
hand-derived expectations.
"""

import pytest

from dashboard_core.mitigation import run_zne_mitigation, run_density_matrix_zne

BELL_QASM = (
    'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; '
    'h q[0]; cx q[0],q[1]; measure q -> c;'
)


class TestRunZneMitigation:

    def test_ideal_state_zz_expectation_is_one(self):
        result = run_zne_mitigation(BELL_QASM, 'ZZ', 'ideal', 0.0, seed=1, n_trials=5)
        assert result.ideal_expectation == pytest.approx(1.0, abs=1e-9)
        assert result.n_qubits == 2
        assert result.pauli_string == 'ZZ'

    def test_no_noise_means_no_noise_at_any_scale(self):
        # noise_p=0.0 short-circuits to the ideal channel at every scale
        # (see dense_evolution.NoiseModel) -- ZNE of a constant is that
        # same constant.
        result = run_zne_mitigation(BELL_QASM, 'ZZ', 'ideal', 0.0, seed=1, n_trials=5)
        assert all(v == pytest.approx(1.0, abs=1e-9) for v in result.noisy_expectations)
        assert result.zne_extrapolated == pytest.approx(1.0, abs=1e-9)

    def test_real_depolarizing_noise_decays_with_scale(self):
        result = run_zne_mitigation(BELL_QASM, 'ZZ', 'depolarizing', 0.1, seed=7, n_trials=100)
        # Real depolarizing channel: <ZZ> should decrease monotonically
        # as the noise scale (1x/2x/3x) increases.
        f1, f2, f3 = result.noisy_expectations
        assert f1 > f2 > f3
        assert f1 == pytest.approx(0.78, abs=0.05)

    def test_same_seed_is_deterministic(self):
        r1 = run_zne_mitigation(BELL_QASM, 'ZZ', 'depolarizing', 0.1, seed=7, n_trials=50)
        r2 = run_zne_mitigation(BELL_QASM, 'ZZ', 'depolarizing', 0.1, seed=7, n_trials=50)
        assert r1.noisy_expectations == r2.noisy_expectations
        assert r1.zne_extrapolated == r2.zne_extrapolated

    def test_pauli_string_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="pauli_string length"):
            run_zne_mitigation(BELL_QASM, 'ZZZ', 'ideal', 0.0)

    def test_unknown_noise_model_raises(self):
        with pytest.raises(ValueError, match="unknown noise model"):
            run_zne_mitigation(BELL_QASM, 'ZZ', 'not_a_real_model', 0.1)


class TestRunDensityMatrixZne:

    def test_ideal_case_fidelity_is_one(self):
        result = run_density_matrix_zne(BELL_QASM, 'ideal', 0.0, seed=1, n_trials=5)
        assert result.fidelity_raw == pytest.approx(1.0, abs=1e-6)
        assert result.fidelity_corrected == pytest.approx(1.0, abs=1e-6)
        assert result.n_qubits == 2

    def test_zne_correction_improves_fidelity_over_raw_noisy(self):
        # Real, honest measurement (never fed back into the extrapolation
        # itself) -- the whole point of grading via Uhlmann fidelity is
        # that this isn't guaranteed to improve for every circuit/noise
        # combination, but for this real case (Bell state, real
        # depolarizing channel) it does.
        result = run_density_matrix_zne(BELL_QASM, 'depolarizing', 0.1, seed=7, n_trials=100)
        assert result.fidelity_corrected > result.fidelity_raw
        assert result.fidelity_raw == pytest.approx(0.815, abs=0.03)
        assert result.fidelity_corrected == pytest.approx(0.958, abs=0.03)

    def test_unknown_noise_model_raises(self):
        with pytest.raises(ValueError, match="unknown noise model"):
            run_density_matrix_zne(BELL_QASM, 'not_a_real_model', 0.1)
