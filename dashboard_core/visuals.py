"""
Circuit-diagram drawing is native (dashboard_core.circuit_diagram, plain
matplotlib, never a Qiskit QuantumCircuit -- see that module's docstring
for why). The other three panels are thin wrappers around Qiskit's own
real visualization functions, which only ever take a statevector array or
counts dict, never a QuantumCircuit -- every function below adapts our
data (produced by dashboard_core.engine) into the shape each expects and
returns the resulting matplotlib Figure unchanged.
"""

import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector, plot_histogram, plot_state_qsphere

from .circuit_diagram import draw_native_circuit_diagram

__all__ = ['draw_circuit_figure', 'histogram_figure', 'qsphere_figure', 'bloch_multivector_figure']

# dense_evolution.registry sets plt.style.use('dark_background') globally
# for its own diagnostic plots (import-time side effect). plot_histogram
# and plot_state_qsphere don't set their own facecolor and so silently
# inherit that ambient rcParams state -- circuit.draw(output='mpl') does
# set its own style and isn't affected, which is why only two of the three
# panels came out dark. Rendering inside style.context('default') pins
# every figure here to matplotlib's light default regardless of whatever
# other modules have globally changed.
_LIGHT_STYLE = 'default'


def draw_circuit_figure(ops, n_qubits: int):
    """Native matplotlib circuit diagram (dashboard_core.circuit_diagram) --
    never builds a Qiskit QuantumCircuit."""
    with plt.style.context(_LIGHT_STYLE):
        return draw_native_circuit_diagram(ops, n_qubits)


def histogram_figure(counts: dict):
    """Qiskit's own shot-count histogram (qiskit.visualization.plot_histogram)."""
    with plt.style.context(_LIGHT_STYLE):
        return plot_histogram(counts)


def qsphere_figure(statevector):
    """Qiskit's own Q-sphere (qiskit.visualization.plot_state_qsphere) --
    the multi-qubit generalization of the Bloch sphere used by IBM Quantum
    Composer."""
    with plt.style.context(_LIGHT_STYLE):
        return plot_state_qsphere(Statevector(statevector))


def bloch_multivector_figure(statevector):
    """Qiskit's own per-qubit Bloch spheres (plot_bloch_multivector)."""
    with plt.style.context(_LIGHT_STYLE):
        return plot_bloch_multivector(Statevector(statevector))
