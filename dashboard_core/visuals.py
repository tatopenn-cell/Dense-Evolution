"""
Thin wrappers around Qiskit's own real visualization functions -- no
hand-rolled circuit drawing or Bloch-sphere plotting here. Every function
below just adapts our data (a Qiskit QuantumCircuit, a statevector array,
a counts dict, all produced by dashboard_core.engine) into the shape
qiskit.visualization already expects, and returns the resulting
matplotlib Figure unchanged.
"""

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector, plot_histogram, plot_state_qsphere

__all__ = ['draw_circuit_figure', 'histogram_figure', 'qsphere_figure', 'bloch_multivector_figure']


def draw_circuit_figure(circuit: QuantumCircuit):
    """Qiskit's own matplotlib circuit diagram (circuit.draw(output='mpl'))."""
    return circuit.draw(output='mpl')


def histogram_figure(counts: dict):
    """Qiskit's own shot-count histogram (qiskit.visualization.plot_histogram)."""
    return plot_histogram(counts)


def qsphere_figure(statevector):
    """Qiskit's own Q-sphere (qiskit.visualization.plot_state_qsphere) --
    the multi-qubit generalization of the Bloch sphere used by IBM Quantum
    Composer."""
    return plot_state_qsphere(Statevector(statevector))


def bloch_multivector_figure(statevector):
    """Qiskit's own per-qubit Bloch spheres (plot_bloch_multivector)."""
    return plot_bloch_multivector(Statevector(statevector))
