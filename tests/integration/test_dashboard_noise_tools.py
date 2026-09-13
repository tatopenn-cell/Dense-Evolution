"""
Tests for dashboard_core.noise_tools -- the standalone noise-profile
wrappers (cosmic-ray burst, oscillating p_eff) and the density-matrix
channel wrapper, none of which had any test coverage before (prog.txt,
dashboard_core audit point 3d).
"""

import pytest

from dashboard_core.noise_tools import (
    run_cosmic_ray_burst, run_oscillating_noise, run_density_matrix_channel,
)

H_QASM = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; creg c[1]; h q[0]; measure q -> c;'
BELL_QASM = (
    'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; '
    'h q[0]; cx q[0],q[1]; measure q -> c;'
)


class TestCosmicRayBurst:

    def test_returns_expected_shape_and_peak(self):
        result = run_cosmic_ray_burst(baseline_gamma=0.01)
        assert len(result.times_us) == len(result.decay_probabilities) == 4
        assert result.peak_ratio >= 1.0

    def test_custom_times(self):
        result = run_cosmic_ray_burst(baseline_gamma=0.02, times_us=[0.0, 5.0])
        assert result.times_us == [0.0, 5.0]
        assert len(result.decay_probabilities) == 2


class TestOscillatingNoise:

    def test_returns_expected_shape(self):
        result = run_oscillating_noise(base_p=0.1, freq=1.0, amp=0.05)
        assert len(result.factors) == len(result.p_eff) == 4

    def test_custom_factors(self):
        result = run_oscillating_noise(base_p=0.1, freq=1.0, amp=0.05, factors=[0.0, 0.5])
        assert result.factors == [0.0, 0.5]
        assert len(result.p_eff) == 2


class TestDensityMatrixChannel:

    def test_invalid_channel_raises(self):
        with pytest.raises(ValueError, match="channel must be"):
            run_density_matrix_channel(H_QASM, 'not_a_real_channel', 0.1)

    def test_amplitude_damping_rejects_multi_qubit(self):
        with pytest.raises(ValueError, match="single-qubit only"):
            run_density_matrix_channel(BELL_QASM, 'amplitude_damping', 0.1)

    def test_amplitude_damping_single_qubit(self):
        result = run_density_matrix_channel(H_QASM, 'amplitude_damping', 0.2)
        assert result.n_qubits == 1
        assert result.trace == pytest.approx(1.0, abs=1e-9)
        # |+> under amplitude damping: P(0) rises above the ideal 0.5
        assert result.noisy_diagonal[0] > result.ideal_diagonal[0]

    def test_global_depolarizing_multi_qubit(self):
        result = run_density_matrix_channel(BELL_QASM, 'global_depolarizing', 0.3)
        assert result.n_qubits == 2
        assert result.trace == pytest.approx(1.0, abs=1e-9)

    def test_zero_param_leaves_diagonal_unchanged(self):
        # BUG FIX regression companion (prog.txt point 3d): SafeMemoryGuard
        # is now checked before allocating -- this just confirms normal,
        # small-circuit usage still works exactly as before that guard
        # was added.
        result = run_density_matrix_channel(H_QASM, 'amplitude_damping', 0.0)
        for ideal, noisy in zip(result.ideal_diagonal, result.noisy_diagonal):
            assert noisy == pytest.approx(ideal, abs=1e-9)
