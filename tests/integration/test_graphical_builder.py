"""
Tests for dashboard_core.graphical_builder.ops_to_native_tuples -- the op
list produced by the drag-and-drop circuit builder component must turn
into dense_evolution's own (name, *qubits[, param]) gate tuples that run
on the exact same engine as typed OpenQASM. No Qiskit QuantumCircuit is
ever built here (see dashboard_core/engine.py's module docstring for why
that matters on macOS), so this file needs no macOS skip at all.
"""

import numpy as np
import pytest

from dashboard_core.engine import run_circuit_from_qasm
from dashboard_core.graphical_builder import ops_to_native_tuples
from dashboard_core.qasm_library import gate_tuples_to_qasm


def test_bell_state_ops_match_bell_qasm():
    ops = [
        {"gate": "h", "qubits": [0]},
        {"gate": "cx", "qubits": [0, 1]},
    ]
    native = ops_to_native_tuples(2, ops)
    qasm_from_grid = gate_tuples_to_qasm(native, 2)

    result_from_grid = run_circuit_from_qasm(qasm_from_grid, n_shots=10, seed=1)
    result_from_bell_qasm = run_circuit_from_qasm(
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
        'h q[0];\ncx q[0],q[1];\nmeasure q -> c;\n',
        n_shots=10, seed=1,
    )
    assert np.allclose(result_from_grid.statevector, result_from_bell_qasm.statevector, atol=1e-9)


def test_single_qubit_gates_placed_correctly():
    ops = [{"gate": "x", "qubits": [1]}]
    native = ops_to_native_tuples(3, ops)
    qasm = gate_tuples_to_qasm(native, 3)
    result = run_circuit_from_qasm(qasm, n_shots=10, seed=1)
    # X on qubit 1 (Qiskit little-endian: qubit 1 is bit index 1) -> state |010> = index 2
    expected = np.zeros(8, dtype=complex)
    expected[0b010] = 1.0
    assert np.allclose(result.statevector, expected, atol=1e-9)


def test_cy_and_cz_gates():
    ops_cz = [{"gate": "h", "qubits": [0]}, {"gate": "cz", "qubits": [0, 1]}]
    native_cz = ops_to_native_tuples(2, ops_cz)
    assert any(op[0] == "cz" for op in native_cz)

    ops_cy = [{"gate": "h", "qubits": [0]}, {"gate": "cy", "qubits": [0, 1]}]
    native_cy = ops_to_native_tuples(2, ops_cy)
    assert any(op[0] == "cy" for op in native_cy)


def test_swap_gate():
    ops = [{"gate": "x", "qubits": [0]}, {"gate": "swap", "qubits": [0, 1]}]
    native = ops_to_native_tuples(2, ops)
    qasm = gate_tuples_to_qasm(native, 2)
    result = run_circuit_from_qasm(qasm, n_shots=10, seed=1)
    # X on qubit 0 then SWAP(0,1) -> qubit 1 ends up excited -> |10> (bit1=1,bit0=0) = index 2
    expected = np.zeros(4, dtype=complex)
    expected[0b10] = 1.0
    assert np.allclose(result.statevector, expected, atol=1e-9)


def test_rotation_gate_applies_a_real_nonzero_rotation():
    ops = [{"gate": "rx", "qubits": [0]}]
    native = ops_to_native_tuples(1, ops)
    qasm = gate_tuples_to_qasm(native, 1)
    result = run_circuit_from_qasm(qasm, n_shots=10, seed=1)
    # Rx(pi/2)|0> is an equal-ish superposition, not |0> or |1> exactly.
    assert not np.allclose(np.abs(result.statevector), [1.0, 0.0], atol=1e-6)
    assert not np.allclose(np.abs(result.statevector), [0.0, 1.0], atol=1e-6)


def test_unknown_gate_raises():
    with pytest.raises(ValueError, match="unknown gate"):
        ops_to_native_tuples(2, [{"gate": "bogus", "qubits": [0]}])


def test_out_of_range_qubit_raises():
    with pytest.raises(ValueError, match="out of range"):
        ops_to_native_tuples(2, [{"gate": "h", "qubits": [5]}])


def test_zero_qubits_raises():
    with pytest.raises(ValueError, match="at least 1 qubit"):
        ops_to_native_tuples(0, [])


def test_empty_ops_is_just_measurement():
    native = ops_to_native_tuples(2, [])
    assert native == []
    qasm = gate_tuples_to_qasm(native, 2)
    assert "measure q -> c;" in qasm
