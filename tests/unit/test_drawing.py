"""
Unit tests for dense_evolution/drawing.py -- draw_circuit.
"""
import pytest

from dense_evolution.drawing import draw_circuit


class TestDrawCircuit:

    def test_one_row_per_qubit(self):
        diagram = draw_circuit([('h', 0), ('cx', 0, 1)], n_qubits=3)
        assert len(diagram.split('\n')) == 3

    def test_rows_start_with_qubit_labels(self):
        diagram = draw_circuit([], n_qubits=2)
        lines = diagram.split('\n')
        assert lines[0].startswith('q0:')
        assert lines[1].startswith('q1:')

    def test_single_qubit_gate_label_appears_on_its_own_row_only(self):
        diagram = draw_circuit([('h', 1)], n_qubits=3)
        lines = diagram.split('\n')
        assert 'H' in lines[1]
        assert 'H' not in lines[0]
        assert 'H' not in lines[2]

    def test_control_and_target_symbols_appear_for_cx(self):
        diagram = draw_circuit([('cx', 0, 2)], n_qubits=3)
        lines = diagram.split('\n')
        assert '*' in lines[0]  # control
        assert 'X' in lines[2]  # target
        assert '|' in lines[1]  # pass-through on the qubit in between

    def test_swap_marks_both_qubits(self):
        diagram = draw_circuit([('swap', 0, 1)], n_qubits=2)
        lines = diagram.split('\n')
        assert 'x' in lines[0] and 'x' in lines[1]

    def test_all_rows_have_equal_length(self):
        diagram = draw_circuit(
            [('h', 0), ('cx', 0, 1), ('rz', 1, 1.23), ('swap', 0, 2)], n_qubits=3)
        lines = diagram.split('\n')
        assert len({len(line) for line in lines}) == 1

    def test_ascii_only_no_unicode_box_drawing(self):
        # a printed diagram must survive a plain ASCII/cp1252 console --
        # regression guard for the Unicode box-drawing chars this used at
        # first, which crash `print()` on a default Windows console.
        diagram = draw_circuit(
            [('h', 0), ('cx', 0, 1), ('swap', 0, 2), ('ccx', 0, 1, 2)], n_qubits=3)
        diagram.encode('ascii')  # raises UnicodeEncodeError if not pure ASCII

    def test_empty_circuit_still_draws_bare_wires(self):
        diagram = draw_circuit([], n_qubits=2)
        assert len(diagram.split('\n')) == 2

    def test_rejects_fewer_than_one_qubit(self):
        with pytest.raises(ValueError):
            draw_circuit([], n_qubits=0)
