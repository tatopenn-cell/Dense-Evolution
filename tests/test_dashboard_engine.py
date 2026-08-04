"""
Tests for dashboard_core.engine.run_circuit_from_qasm -- verifies the
real wiring (QASM -> dense_evolution's own QASMParser ->
dense_evolution.DenseSVSimulator) against known-exact circuits, not
against a mock.

Reference statevectors below are the textbook analytic results (Bell/GHZ
are symmetric under bit-order, so Qiskit's little-endian convention and
dense_evolution's native MSB-first convention agree here without needing
a Qiskit Statevector cross-check) -- no Qiskit involved in this file at
all, matching run_circuit_from_qasm itself never constructing a
qiskit.circuit.QuantumCircuit (see dashboard_core/engine.py's module
docstring for why that matters on macOS).
"""

import numpy as np
import pytest

from dashboard_core.engine import run_circuit_from_qasm

_INV_SQRT2 = 1 / np.sqrt(2)

BELL_QASM = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
    'qreg q[2];\ncreg c[2];\n'
    'h q[0];\ncx q[0],q[1];\n'
    'measure q -> c;\n'
)

GHZ_QASM = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
    'qreg q[3];\ncreg c[3];\n'
    'h q[0];\ncx q[0],q[1];\ncx q[1],q[2];\n'
    'measure q -> c;\n'
)


def test_bell_state_statevector_matches_analytic_reference():
    result = run_circuit_from_qasm(BELL_QASM, n_shots=10, seed=1)
    expected = np.array([_INV_SQRT2, 0, 0, _INV_SQRT2], dtype=complex)
    assert np.allclose(result.statevector, expected, atol=1e-9)


def test_bell_state_probabilities_are_50_50_on_00_and_11():
    result = run_circuit_from_qasm(BELL_QASM, n_shots=10, seed=1)
    assert result.probabilities[0] == pytest.approx(0.5, abs=1e-9)
    assert result.probabilities[3] == pytest.approx(0.5, abs=1e-9)
    assert result.probabilities[1] == pytest.approx(0.0, abs=1e-9)
    assert result.probabilities[2] == pytest.approx(0.0, abs=1e-9)


def test_bell_state_counts_only_contain_00_and_11():
    result = run_circuit_from_qasm(BELL_QASM, n_shots=500, seed=7)
    assert sum(result.counts.values()) == 500
    assert set(result.counts.keys()) <= {"00", "11"}


def test_ghz_state_matches_analytic_reference():
    result = run_circuit_from_qasm(GHZ_QASM, n_shots=10, seed=3)
    expected = np.zeros(8, dtype=complex)
    expected[0] = _INV_SQRT2
    expected[7] = _INV_SQRT2
    assert np.allclose(result.statevector, expected, atol=1e-9)
    assert result.n_qubits == 3


def test_seed_gives_reproducible_counts():
    r1 = run_circuit_from_qasm(BELL_QASM, n_shots=200, seed=99)
    r2 = run_circuit_from_qasm(BELL_QASM, n_shots=200, seed=99)
    assert r1.counts == r2.counts


def test_zero_qubit_circuit_raises():
    with pytest.raises(ValueError, match="at least 1 qubit"):
        run_circuit_from_qasm(
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[0];\ncreg c[0];\n', n_shots=10,
        )


def test_ideal_run_has_no_fidelity_vs_ideal():
    result = run_circuit_from_qasm(BELL_QASM, n_shots=10, seed=1)
    assert result.fidelity_vs_ideal is None


def test_zero_noise_probability_has_no_fidelity_vs_ideal():
    result = run_circuit_from_qasm(BELL_QASM, n_shots=10, seed=1, noise_model="depolarizing", noise_p=0.0)
    assert result.fidelity_vs_ideal is None


def test_noisy_run_fidelity_vs_ideal_is_a_valid_probability():
    result = run_circuit_from_qasm(BELL_QASM, n_shots=10, seed=1, noise_model="depolarizing", noise_p=0.3)
    assert 0.0 <= result.fidelity_vs_ideal <= 1.0 + 1e-9


def test_noisy_run_fidelity_vs_ideal_matches_direct_overlap_computation():
    # Independent check, not just "some fidelity function ran": the ideal
    # statevector doesn't depend on noise_p/rng at all, so a separate
    # noise_model="ideal" call with the same seed must reproduce the exact
    # pre-noise state the noisy run compared itself against -- then
    # |<ideal|noisy>|^2 computed here from the two returned statevectors
    # (same Qiskit bit-order convention, so a plain inner product is valid)
    # must match what engine.py reported.
    ideal = run_circuit_from_qasm(BELL_QASM, n_shots=10, seed=1)
    noisy = run_circuit_from_qasm(BELL_QASM, n_shots=10, seed=1, noise_model="depolarizing", noise_p=0.3)
    expected = abs(np.vdot(ideal.statevector, noisy.statevector)) ** 2
    assert noisy.fidelity_vs_ideal == pytest.approx(expected, abs=1e-9)
