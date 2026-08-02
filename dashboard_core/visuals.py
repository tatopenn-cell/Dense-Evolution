"""
Thin wrappers around Qiskit's own real visualization functions -- no
hand-rolled circuit drawing or Bloch-sphere plotting here. Every function
below just adapts our data (a Qiskit QuantumCircuit, a statevector array,
a counts dict, all produced by dashboard_core.engine) into the shape
qiskit.visualization already expects, and returns the resulting
matplotlib Figure unchanged.
"""

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector, plot_histogram, plot_state_qsphere

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


def draw_circuit_figure(circuit: QuantumCircuit):
    """Qiskit's own matplotlib circuit diagram (circuit.draw(output='mpl'))."""
    with plt.style.context(_LIGHT_STYLE):
        return circuit.draw(output='mpl')


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
