"""
Tests for dashboard_core.visuals -- each wrapper must return a real
matplotlib Figure produced by Qiskit's own visualization functions, for
data actually coming out of dashboard_core.engine.
"""

import matplotlib
matplotlib.use('Agg')

from matplotlib.figure import Figure

from dashboard_core.engine import run_circuit_from_qasm
from dashboard_core.visuals import (
    bloch_multivector_figure, draw_circuit_figure, histogram_figure, qsphere_figure,
)

BELL_QASM = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
    'qreg q[2];\ncreg c[2];\n'
    'h q[0];\ncx q[0],q[1];\n'
    'measure q -> c;\n'
)


def _bell_result():
    return run_circuit_from_qasm(BELL_QASM, n_shots=200, seed=5)


def test_draw_circuit_figure_returns_a_figure():
    result = _bell_result()
    fig = draw_circuit_figure(result.qiskit_circuit)
    assert isinstance(fig, Figure)


def test_histogram_figure_returns_a_figure():
    result = _bell_result()
    fig = histogram_figure(result.counts)
    assert isinstance(fig, Figure)


def test_qsphere_figure_returns_a_figure():
    result = _bell_result()
    fig = qsphere_figure(result.statevector)
    assert isinstance(fig, Figure)


def test_bloch_multivector_figure_returns_a_figure():
    result = _bell_result()
    fig = bloch_multivector_figure(result.statevector)
    assert isinstance(fig, Figure)
