"""
Tests for dashboard_core.engine.run_circuit_from_qasm -- verifies the
real wiring (QASM -> Qiskit circuit -> dense_evolution.DenseSVSimulator)
against known-exact circuits, not against a mock.
"""

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from dashboard_core.engine import run_circuit_from_qasm

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


def test_bell_state_statevector_matches_qiskit_reference():
    result = run_circuit_from_qasm(BELL_QASM, n_shots=10, seed=1)
    circuit_without_measurements = result.qiskit_circuit.remove_final_measurements(inplace=False)
    expected = Statevector(circuit_without_measurements).data
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


def test_ghz_state_matches_qiskit_reference():
    result = run_circuit_from_qasm(GHZ_QASM, n_shots=10, seed=3)
    circuit_without_measurements = result.qiskit_circuit.remove_final_measurements(inplace=False)
    expected = Statevector(circuit_without_measurements).data
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
