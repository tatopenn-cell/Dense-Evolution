"""
Tests for dashboard_core.graphical_builder.ops_to_qiskit_circuit -- the
op list produced by the drag-and-drop circuit builder component must turn
into a real, correctly-wired QuantumCircuit that runs on the same engine
as typed OpenQASM.
"""

import sys

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from dashboard_core.engine import run_circuit_from_qasm
from dashboard_core.graphical_builder import ops_to_qiskit_circuit
from qiskit import qasm2

# Same known upstream Qiskit bug as test_interop.py::TestQiskitInterop
# and test_dashboard_visuals.py (see either for the full story):
# QuantumCircuit.__init__ segfaults on macOS CI runners on its own,
# independent of Dense-Evolution/dashboard_core. ops_to_qiskit_circuit
# and run_circuit_from_qasm both build one, so every test here hits it.
pytestmark = pytest.mark.skipif(
    sys.platform == 'darwin',
    reason="qiskit.circuit.QuantumCircuit.__init__ segfaults on macOS CI -- see test_interop.py::TestQiskitInterop",
)


def test_bell_state_ops_match_bell_qasm():
    ops = [
        {"gate": "h", "qubits": [0]},
        {"gate": "cx", "qubits": [0, 1]},
    ]
    qc = ops_to_qiskit_circuit(2, ops)
    qasm_from_grid = qasm2.dumps(qc)

    result_from_grid = run_circuit_from_qasm(qasm_from_grid, n_shots=10, seed=1)
    result_from_bell_qasm = run_circuit_from_qasm(
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
        'h q[0];\ncx q[0],q[1];\nmeasure q -> c;\n',
        n_shots=10, seed=1,
    )
    assert np.allclose(result_from_grid.statevector, result_from_bell_qasm.statevector, atol=1e-9)


def test_single_qubit_gates_placed_correctly():
    ops = [{"gate": "x", "qubits": [1]}]
    qc = ops_to_qiskit_circuit(3, ops)
    circuit_without_measurements = qc.remove_final_measurements(inplace=False)
    sv = Statevector(circuit_without_measurements).data
    # X on qubit 1 (Qiskit little-endian: qubit 1 is bit index 1) -> state |010> = index 2
    expected = np.zeros(8, dtype=complex)
    expected[0b010] = 1.0
    assert np.allclose(sv, expected, atol=1e-9)


def test_cy_and_cz_gates():
    ops_cz = [{"gate": "h", "qubits": [0]}, {"gate": "cz", "qubits": [0, 1]}]
    qc_cz = ops_to_qiskit_circuit(2, ops_cz)
    assert qc_cz.count_ops().get("cz") == 1

    ops_cy = [{"gate": "h", "qubits": [0]}, {"gate": "cy", "qubits": [0, 1]}]
    qc_cy = ops_to_qiskit_circuit(2, ops_cy)
    assert qc_cy.count_ops().get("cy") == 1


def test_swap_gate():
    ops = [{"gate": "x", "qubits": [0]}, {"gate": "swap", "qubits": [0, 1]}]
    qc = ops_to_qiskit_circuit(2, ops)
    circuit_without_measurements = qc.remove_final_measurements(inplace=False)
    sv = Statevector(circuit_without_measurements).data
    # X on qubit 0 then SWAP(0,1) -> qubit 1 ends up excited -> |10> (bit1=1,bit0=0) = index 2
    expected = np.zeros(4, dtype=complex)
    expected[0b10] = 1.0
    assert np.allclose(sv, expected, atol=1e-9)


def test_rotation_gate_applies_a_real_nonzero_rotation():
    ops = [{"gate": "rx", "qubits": [0]}]
    qc = ops_to_qiskit_circuit(1, ops)
    circuit_without_measurements = qc.remove_final_measurements(inplace=False)
    sv = Statevector(circuit_without_measurements).data
    # Rx(pi/2)|0> is an equal-ish superposition, not |0> or |1> exactly.
    assert not np.allclose(np.abs(sv), [1.0, 0.0], atol=1e-6)
    assert not np.allclose(np.abs(sv), [0.0, 1.0], atol=1e-6)


def test_unknown_gate_raises():
    with pytest.raises(ValueError, match="unknown gate"):
        ops_to_qiskit_circuit(2, [{"gate": "bogus", "qubits": [0]}])


def test_out_of_range_qubit_raises():
    with pytest.raises(ValueError, match="out of range"):
        ops_to_qiskit_circuit(2, [{"gate": "h", "qubits": [5]}])


def test_zero_qubits_raises():
    with pytest.raises(ValueError, match="at least 1 qubit"):
        ops_to_qiskit_circuit(0, [])


def test_empty_ops_is_just_measurement():
    qc = ops_to_qiskit_circuit(2, [])
    assert qc.count_ops().get("measure") == 2
