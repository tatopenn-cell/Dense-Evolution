"""
Tests for dashboard_core.visuals -- every panel (circuit diagram,
histogram, Q-sphere, Bloch spheres) is native matplotlib/numpy, no Qiskit
anywhere (see dashboard_core/circuit_diagram.py and state_visuals.py's
own docstrings for why that mattered on macOS). Beyond "returns a real
Figure", the state_visuals tests below check the actual physics: a Bell
pair's reduced single-qubit states must come back maximally mixed (a
known analytic fact about maximal entanglement), not just "no exception".
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest

from matplotlib.figure import Figure

from dashboard_core.engine import run_circuit_from_qasm
from dashboard_core.visuals import (
    bloch_multivector_figure, draw_circuit_figure, histogram_figure, qsphere_figure,
)
from dashboard_core.state_visuals import _reduced_density_matrix, _bloch_vector

BELL_QASM = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
    'qreg q[2];\ncreg c[2];\n'
    'h q[0];\ncx q[0],q[1];\n'
    'measure q -> c;\n'
)


@pytest.fixture(autouse=True)
def _close_figures_after_each_test():
    # visuals.py's functions never call plt.close() themselves (they
    # return the Figure for the caller to use), so pyplot's global figure
    # registry would otherwise accumulate one live Figure per test in
    # this file for the rest of the pytest session -- real cleanup
    # hygiene on its own merits, independent of the macOS skip above.
    yield
    plt.close('all')


def _bell_result():
    return run_circuit_from_qasm(BELL_QASM, n_shots=200, seed=5)


def test_draw_circuit_figure_returns_a_figure():
    result = _bell_result()
    fig = draw_circuit_figure(result.ops, result.n_qubits)
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


class TestReducedDensityMatrixAndBlochVector:
    """dashboard_core.state_visuals' own partial-trace math, checked
    against known analytic states -- the same real-physics standard the
    rest of this project holds every other function to."""

    def test_pure_zero_state_bloch_vector_is_north_pole(self):
        sv = np.array([1, 0], dtype=complex)
        rho = _reduced_density_matrix(sv, 1, 0)
        assert np.allclose(_bloch_vector(rho), [0, 0, 1], atol=1e-9)

    def test_pure_one_state_bloch_vector_is_south_pole(self):
        sv = np.array([0, 1], dtype=complex)
        rho = _reduced_density_matrix(sv, 1, 0)
        assert np.allclose(_bloch_vector(rho), [0, 0, -1], atol=1e-9)

    def test_plus_state_bloch_vector_is_on_the_equator(self):
        sv = np.array([1, 1], dtype=complex) / np.sqrt(2)
        rho = _reduced_density_matrix(sv, 1, 0)
        assert np.allclose(_bloch_vector(rho), [1, 0, 0], atol=1e-9)

    def test_bell_pair_each_qubit_is_maximally_mixed(self):
        # The defining feature of maximal entanglement: neither qubit has
        # a well-defined individual state, so its reduced density matrix
        # is exactly I/2 and its Bloch vector has zero length.
        sv = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        for qubit in (0, 1):
            rho = _reduced_density_matrix(sv, 2, qubit)
            assert np.allclose(rho, np.eye(2) / 2, atol=1e-9)
            assert np.linalg.norm(_bloch_vector(rho)) == pytest.approx(0.0, abs=1e-9)

    def test_ghz3_each_qubit_is_maximally_mixed(self):
        sv = np.zeros(8, dtype=complex)
        sv[0] = sv[7] = 1 / np.sqrt(2)
        for qubit in (0, 1, 2):
            rho = _reduced_density_matrix(sv, 3, qubit)
            assert np.allclose(rho, np.eye(2) / 2, atol=1e-9)
