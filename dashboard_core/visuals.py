"""
Every visualization here is native (plain matplotlib/numpy) -- no Qiskit
anywhere in this module. The circuit diagram was already native
(dashboard_core.circuit_diagram); histogram/Q-sphere/Bloch spheres are
now native too (dashboard_core.state_visuals), replacing
qiskit.visualization.{plot_histogram, plot_state_qsphere,
plot_bloch_multivector} -- the last Qiskit call sites this dashboard had.
Qiskit's macOS instability was never proven scoped to just
QuantumCircuit (see circuit_diagram.py's own docstring), so making these
three merely "optional" instead of actually removing Qiskit would have
left the same open question for a different set of functions; there's no
reason left to keep it as a dependency here at all.
"""

import matplotlib.pyplot as plt

from .circuit_diagram import draw_native_circuit_diagram
from .state_visuals import native_histogram_figure, native_qsphere_figure, native_bloch_multivector_figure

__all__ = ['draw_circuit_figure', 'histogram_figure', 'qsphere_figure', 'bloch_multivector_figure']

# dense_evolution.registry sets plt.style.use('dark_background') globally
# for its own diagnostic plots (import-time side effect). Rendering inside
# style.context('default') pins every figure here to matplotlib's light
# default regardless of whatever other modules have globally changed.
_LIGHT_STYLE = 'default'


def draw_circuit_figure(ops, n_qubits: int):
    """Native matplotlib circuit diagram (dashboard_core.circuit_diagram)."""
    with plt.style.context(_LIGHT_STYLE):
        return draw_native_circuit_diagram(ops, n_qubits)


def histogram_figure(counts: dict):
    """Native shot-count histogram (dashboard_core.state_visuals)."""
    with plt.style.context(_LIGHT_STYLE):
        return native_histogram_figure(counts)


def qsphere_figure(statevector):
    """Native Q-sphere (dashboard_core.state_visuals) -- the multi-qubit
    generalization of the Bloch sphere used by IBM Quantum Composer."""
    with plt.style.context(_LIGHT_STYLE):
        return native_qsphere_figure(statevector)


def bloch_multivector_figure(statevector):
    """Native per-qubit Bloch spheres (dashboard_core.state_visuals)."""
    with plt.style.context(_LIGHT_STYLE):
        return native_bloch_multivector_figure(statevector)
