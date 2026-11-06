"""Tests of the input values of the qubits of an OpenQASM text."""
import pytest
from dashboard_core.input_values import qubits_in_qasm, with_input_values

BELL = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
    "h q[0];\ncx q[0],q[1];\nmeasure q -> c;\n"
)
HEAD = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
TAIL = "h q[0];\ncx q[0],q[1];\nmeasure q -> c;\n"


def test_x_line_goes_after_the_declarations_and_before_the_gates():
    assert with_input_values(BELL, [1]) == HEAD + "x q[1];\n" + TAIL


def test_no_qubit_returns_the_text_unchanged():
    assert with_input_values(BELL, []) == BELL


def test_lines_are_in_increasing_order_of_the_index():
    assert with_input_values(BELL, [1, 0]) == HEAD + "x q[0];\nx q[1];\n" + TAIL


def test_an_index_out_of_range_is_ignored():
    assert with_input_values(BELL, [5]) == BELL


def test_a_negative_index_and_a_text_with_no_qreg_are_refused():
    with pytest.raises(ValueError, match="-1"):
        with_input_values(BELL, [-1])
    with pytest.raises(ValueError, match="qreg"):
        with_input_values("OPENQASM 2.0;\n", [0])


def test_qubits_in_qasm():
    assert qubits_in_qasm("OPENQASM 2.0;\nqreg q[2];\n") == 2
    assert qubits_in_qasm("OPENQASM 2.0;\n") == 0
